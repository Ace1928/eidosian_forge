from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1FunctionResponse(_messages.Message):
    """The result output from a [FunctionCall] that contains a string
  representing the [FunctionDeclaration.name] and a structured JSON object
  containing any output from the function is used as context to the model.
  This should contain the result of a [FunctionCall] made based on model
  prediction.

  Messages:
    ResponseValue: Required. The function response in JSON object format.

  Fields:
    name: Required. The name of the function to call. Matches
      [FunctionDeclaration.name] and [FunctionCall.name].
    response: Required. The function response in JSON object format.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ResponseValue(_messages.Message):
        """Required. The function response in JSON object format.

    Messages:
      AdditionalProperty: An additional property for a ResponseValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ResponseValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    name = _messages.StringField(1)
    response = _messages.MessageField('ResponseValue', 2)