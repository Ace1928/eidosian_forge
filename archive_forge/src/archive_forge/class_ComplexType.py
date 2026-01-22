from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComplexType(_messages.Message):
    """A complex type resource. A complex type describes the schema for asset
  metadata.

  Messages:
    FieldsValue: Mapping of a field name to its type.
    LabelsValue: The labels associated with this resource. Each label is a
      key-value pair.

  Fields:
    allowUndefinedFields: Allow fields that aren't in complex type schema as
      defined in complex type fields.
    createTime: Output only. The creation time.
    fields: Mapping of a field name to its type.
    labels: The labels associated with this resource. Each label is a key-
      value pair.
    name: The resource name of the complex type, in the following form:
      `projects/{project}/locations/{location}/complexTypes/{type}`. Here
      {type} is a resource id. Detailed rules for a resource id are: 1. 1
      character minimum, 63 characters maximum 2. only contains letters,
      digits, underscore and hyphen 3. starts with a letter if length == 1,
      starts with a letter or underscore if length > 1
    updateTime: Output only. The last-modified time.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class FieldsValue(_messages.Message):
        """Mapping of a field name to its type.

    Messages:
      AdditionalProperty: An additional property for a FieldsValue object.

    Fields:
      additionalProperties: Additional properties of type FieldsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a FieldsValue object.

      Fields:
        key: Name of the additional property.
        value: A ComplexFieldType attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('ComplexFieldType', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The labels associated with this resource. Each label is a key-value
    pair.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    allowUndefinedFields = _messages.BooleanField(1)
    createTime = _messages.StringField(2)
    fields = _messages.MessageField('FieldsValue', 3)
    labels = _messages.MessageField('LabelsValue', 4)
    name = _messages.StringField(5)
    updateTime = _messages.StringField(6)