from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SummarizationHelpfulnessInput(_messages.Message):
    """Input for summarization helpfulness metric.

  Fields:
    instance: Required. Summarization helpfulness instance.
    metricSpec: Required. Spec for summarization helpfulness score metric.
  """
    instance = _messages.MessageField('GoogleCloudAiplatformV1beta1SummarizationHelpfulnessInstance', 1)
    metricSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1SummarizationHelpfulnessSpec', 2)