from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1Aspect(_messages.Message):
    """An aspect is a single piece of metadata describing an entry.

  Messages:
    DataValue: Required. The content of the aspect, according to its aspect
      type schema. This will replace content. The maximum size of the field is
      120KB (encoded as UTF-8).

  Fields:
    aspectSource: A GoogleCloudDataplexV1AspectSource attribute.
    aspectType: Output only. The resource name of the type used to create this
      Aspect.
    createTime: Output only. The time when the Aspect was created.
    data: Required. The content of the aspect, according to its aspect type
      schema. This will replace content. The maximum size of the field is
      120KB (encoded as UTF-8).
    path: Output only. The path in the entry under which the aspect is
      attached.
    updateTime: Output only. The time when the Aspect was last updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DataValue(_messages.Message):
        """Required. The content of the aspect, according to its aspect type
    schema. This will replace content. The maximum size of the field is 120KB
    (encoded as UTF-8).

    Messages:
      AdditionalProperty: An additional property for a DataValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DataValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    aspectSource = _messages.MessageField('GoogleCloudDataplexV1AspectSource', 1)
    aspectType = _messages.StringField(2)
    createTime = _messages.StringField(3)
    data = _messages.MessageField('DataValue', 4)
    path = _messages.StringField(5)
    updateTime = _messages.StringField(6)