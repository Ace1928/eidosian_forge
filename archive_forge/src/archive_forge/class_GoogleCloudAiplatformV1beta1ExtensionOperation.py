from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ExtensionOperation(_messages.Message):
    """Operation of an extension.

  Fields:
    functionDeclaration: Output only. Structured representation of a function
      declaration as defined by the OpenAPI Spec.
    operationId: Operation ID that uniquely identifies the operations among
      the extension. See: "Operation Object" in
      https://swagger.io/specification/. This field is parsed from the OpenAPI
      spec. For HTTP extensions, if it does not exist in the spec, we will
      generate one from the HTTP method and path.
  """
    functionDeclaration = _messages.MessageField('GoogleCloudAiplatformV1beta1FunctionDeclaration', 1)
    operationId = _messages.StringField(2)