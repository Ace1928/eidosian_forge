from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1RougeInput(_messages.Message):
    """Input for rouge metric.

  Fields:
    instances: Required. Repeated rouge instances.
    metricSpec: Required. Spec for rouge score metric.
  """
    instances = _messages.MessageField('GoogleCloudAiplatformV1beta1RougeInstance', 1, repeated=True)
    metricSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1RougeSpec', 2)