from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ExecuteExtensionResponse(_messages.Message):
    """Response message for ExtensionExecutionService.ExecuteExtension.

  Fields:
    content: Response content from the extension. The content should be
      conformant to the response.content schema in the extension's
      manifest/OpenAPI spec.
  """
    content = _messages.StringField(1)