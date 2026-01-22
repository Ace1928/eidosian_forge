from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsSummarizationEvaluationMetrics(_messages.Message):
    """A GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsSummarizationE
  valuationMetrics object.

  Fields:
    rougeLSum: ROUGE-L (Longest Common Subsequence) scoring at summary level.
  """
    rougeLSum = _messages.FloatField(1, variant=_messages.Variant.FLOAT)