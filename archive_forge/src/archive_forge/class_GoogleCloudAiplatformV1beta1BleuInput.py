from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1BleuInput(_messages.Message):
    """Input for bleu metric.

  Fields:
    instances: Required. Repeated bleu instances.
    metricSpec: Required. Spec for bleu score metric.
  """
    instances = _messages.MessageField('GoogleCloudAiplatformV1beta1BleuInstance', 1, repeated=True)
    metricSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1BleuSpec', 2)