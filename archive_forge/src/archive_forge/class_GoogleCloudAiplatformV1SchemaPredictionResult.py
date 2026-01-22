from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaPredictionResult(_messages.Message):
    """Represents a line of JSONL in the batch prediction output file.

  Messages:
    InstanceValue: User's input instance. Struct is used here instead of Any
      so that JsonFormat does not append an extra "@type" field when we
      convert the proto to JSON.

  Fields:
    error: The error result. Do not set prediction if this is set.
    instance: User's input instance. Struct is used here instead of Any so
      that JsonFormat does not append an extra "@type" field when we convert
      the proto to JSON.
    key: Optional user-provided key from the input instance.
    prediction: The prediction result. Value is used here instead of Any so
      that JsonFormat does not append an extra "@type" field when we convert
      the proto to JSON and so we can represent array of objects. Do not set
      error if this is set.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class InstanceValue(_messages.Message):
        """User's input instance. Struct is used here instead of Any so that
    JsonFormat does not append an extra "@type" field when we convert the
    proto to JSON.

    Messages:
      AdditionalProperty: An additional property for a InstanceValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a InstanceValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    error = _messages.MessageField('GoogleCloudAiplatformV1SchemaPredictionResultError', 1)
    instance = _messages.MessageField('InstanceValue', 2)
    key = _messages.StringField(3)
    prediction = _messages.MessageField('extra_types.JsonValue', 4)