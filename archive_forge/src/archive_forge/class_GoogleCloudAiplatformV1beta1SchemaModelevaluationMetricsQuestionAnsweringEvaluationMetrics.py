from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsQuestionAnsweringEvaluationMetrics(_messages.Message):
    """A GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsQuestionAnswer
  ingEvaluationMetrics object.

  Fields:
    exactMatch: The rate at which the input predicted strings exactly match
      their references.
  """
    exactMatch = _messages.FloatField(1, variant=_messages.Variant.FLOAT)