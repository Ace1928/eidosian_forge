from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1PairwiseQuestionAnsweringQualityInput(_messages.Message):
    """Input for pairwise question answering quality metric.

  Fields:
    instance: Required. Pairwise question answering quality instance.
    metricSpec: Required. Spec for pairwise question answering quality score
      metric.
  """
    instance = _messages.MessageField('GoogleCloudAiplatformV1beta1PairwiseQuestionAnsweringQualityInstance', 1)
    metricSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1PairwiseQuestionAnsweringQualitySpec', 2)