from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsPairwiseTextGenerationEvaluationMetrics(_messages.Message):
    """Metrics for general pairwise text generation evaluation results.

  Fields:
    accuracy: Fraction of cases where the autorater agreed with the human
      raters.
    baselineModelWinRate: Percentage of time the autorater decided the
      baseline model had the better response.
    cohensKappa: A measurement of agreement between the autorater and human
      raters that takes the likelihood of random agreement into account.
    f1Score: Harmonic mean of precision and recall.
    falseNegativeCount: Number of examples where the autorater chose the
      baseline model, but humans preferred the model.
    falsePositiveCount: Number of examples where the autorater chose the
      model, but humans preferred the baseline model.
    humanPreferenceBaselineModelWinRate: Percentage of time humans decided the
      baseline model had the better response.
    humanPreferenceModelWinRate: Percentage of time humans decided the model
      had the better response.
    modelWinRate: Percentage of time the autorater decided the model had the
      better response.
    precision: Fraction of cases where the autorater and humans thought the
      model had a better response out of all cases where the autorater thought
      the model had a better response. True positive divided by all positive.
    recall: Fraction of cases where the autorater and humans thought the model
      had a better response out of all cases where the humans thought the
      model had a better response.
    trueNegativeCount: Number of examples where both the autorater and humans
      decided that the model had the worse response.
    truePositiveCount: Number of examples where both the autorater and humans
      decided that the model had the better response.
  """
    accuracy = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    baselineModelWinRate = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    cohensKappa = _messages.FloatField(3, variant=_messages.Variant.FLOAT)
    f1Score = _messages.FloatField(4, variant=_messages.Variant.FLOAT)
    falseNegativeCount = _messages.IntegerField(5)
    falsePositiveCount = _messages.IntegerField(6)
    humanPreferenceBaselineModelWinRate = _messages.FloatField(7, variant=_messages.Variant.FLOAT)
    humanPreferenceModelWinRate = _messages.FloatField(8, variant=_messages.Variant.FLOAT)
    modelWinRate = _messages.FloatField(9, variant=_messages.Variant.FLOAT)
    precision = _messages.FloatField(10, variant=_messages.Variant.FLOAT)
    recall = _messages.FloatField(11, variant=_messages.Variant.FLOAT)
    trueNegativeCount = _messages.IntegerField(12)
    truePositiveCount = _messages.IntegerField(13)