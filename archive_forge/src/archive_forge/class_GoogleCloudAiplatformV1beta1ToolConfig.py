from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ToolConfig(_messages.Message):
    """Tool config. This config is shared for all tools provided in the
  request.

  Fields:
    functionCallingConfig: Optional. Function calling config.
  """
    functionCallingConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1FunctionCallingConfig', 1)