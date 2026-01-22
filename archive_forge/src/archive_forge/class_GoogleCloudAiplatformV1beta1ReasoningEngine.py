from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ReasoningEngine(_messages.Message):
    """ReasoningEngine provides a customizable runtime for models to determine
  which actions to take and in which order.

  Fields:
    createTime: Output only. Timestamp when this ReasoningEngine was created.
    description: Optional. The description of the ReasoningEngine.
    displayName: Required. The display name of the ReasoningEngine.
    etag: Optional. Used to perform consistent read-modify-write updates. If
      not set, a blind "overwrite" update happens.
    name: Identifier. The resource name of the ReasoningEngine.
    spec: Required. Configurations of the ReasoningEngine
    updateTime: Output only. Timestamp when this ReasoningEngine was most
      recently updated.
  """
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    etag = _messages.StringField(4)
    name = _messages.StringField(5)
    spec = _messages.MessageField('GoogleCloudAiplatformV1beta1ReasoningEngineSpec', 6)
    updateTime = _messages.StringField(7)