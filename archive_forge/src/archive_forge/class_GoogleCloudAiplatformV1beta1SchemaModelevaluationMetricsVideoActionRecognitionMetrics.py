from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsVideoActionRecognitionMetrics(_messages.Message):
    """Model evaluation metrics for video action recognition.

  Fields:
    evaluatedActionCount: The number of ground truth actions used to create
      this evaluation.
    videoActionMetrics: The metric entries for precision window lengths:
      1s,2s,3s.
  """
    evaluatedActionCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    videoActionMetrics = _messages.MessageField('GoogleCloudAiplatformV1beta1SchemaModelevaluationMetricsVideoActionMetrics', 2, repeated=True)