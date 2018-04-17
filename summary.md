Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs
===
- Varun Gulshan et al.
- JAMA. 2016;316(22):2402-2410. doi:10.1001/jama.2016.17216
- December 13, 2016
- https://jamanetwork.com/journals/jama/fullarticle/2588763

# どんなもの？
眼底写真を用いた糖尿病性網膜症の診断を、deep learningアルゴリズムで自動化。2種類のデータセットでの実験を通じて、高精度で検知できると分かった。

# 先行研究と比べてどこがすごい？
（データセットが違うので単純な比較はできないが）sensitibityとspecificityが高い。

# 技術や手法のキモはどこ？
deep convolutional neural network
- 糖尿病性網膜症の診断に用いられた眼底写真128175枚を訓練データに
- 写真は3~7人の医師や研修医が診断し、多数決でラベル付け
- 出力は0~1の値（糖尿病性網膜症でありそうな度合い）

# どうやって有効だと検証した？
- 2種類のデータセットで、sensitibityとspecificityを調べた。両指標とも約90%以上の値が出た。
- 以下引用(https://en.wikipedia.org/wiki/Sensitivity_and_specificity)

> Sensitivity and specificity are statistical measures of the performance of a binary classification test, also known in statistics as classification function:
> - Sensitivity (also called the true positive rate, the recall, or probability of detection[1] in some fields) measures the proportion of positives that are correctly identified as such (e.g. the percentage of sick people who are correctly identified as having the condition).
> - Specificity (also called the true negative rate) measures the proportion of negatives that are correctly identified as such (e.g. the percentage of healthy people who are correctly identified as not having the condition).

# 議論はある？
- 同システムの利点
  1. consistency of interpretation
  1. high sensitivity and specificity
  1. instantaneous reporting of results
  1. sensitivity and specificityを目的に応じて調整できる
- 今後の展望
  1. より豊富な訓練データ
  1. より多角的なシステム評価
- システムの限界
  1. 医師の判断に基づきラベル付けしているため医師が見つけられないものは見つけられない
  1. neural newworkのブラックボックス性
    - "Hence, this algorithm is not a replacement for a comprehensive eye examination"

# 次に読むべき論文は？
NULL
