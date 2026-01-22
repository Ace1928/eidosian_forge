from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1QueryExtensionRequest(_messages.Message):
    """Request message for ExtensionExecutionService.QueryExtension.

  Fields:
    contents: Required. The content of the current conversation with the
      model. For single-turn queries, this is a single instance. For multi-
      turn queries, this is a repeated field that contains conversation
      history + latest request.
  """
    contents = _messages.MessageField('GoogleCloudAiplatformV1beta1Content', 1, repeated=True)