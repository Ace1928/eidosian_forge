from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class JSONSchemaProps(_messages.Message):
    """JSONSchemaProps is a JSON-Schema following Specification Draft 4
  (http://json-schema.org/).

  Messages:
    DefinitionsValue: A DefinitionsValue object.
    DependenciesValue: A DependenciesValue object.
    PatternPropertiesValue: A PatternPropertiesValue object.
    PropertiesValue: A PropertiesValue object.

  Fields:
    additionalItems: A JSONSchemaPropsOrBool attribute.
    additionalProperties: A JSONSchemaPropsOrBool attribute.
    allOf: A JSONSchemaProps attribute.
    anyOf: A JSONSchemaProps attribute.
    default: A JSON attribute.
    definitions: A DefinitionsValue attribute.
    dependencies: A DependenciesValue attribute.
    description: A string attribute.
    enum: A string attribute.
    example: A JSON attribute.
    exclusiveMaximum: A boolean attribute.
    exclusiveMinimum: A boolean attribute.
    externalDocs: A ExternalDocumentation attribute.
    format: A string attribute.
    id: A string attribute.
    items: A JSONSchemaPropsOrArray attribute.
    maxItems: A string attribute.
    maxLength: A string attribute.
    maxProperties: A string attribute.
    maximum: A number attribute.
    minItems: A string attribute.
    minLength: A string attribute.
    minProperties: A string attribute.
    minimum: A number attribute.
    multipleOf: A number attribute.
    not_: A JSONSchemaProps attribute.
    oneOf: A JSONSchemaProps attribute.
    pattern: A string attribute.
    patternProperties: A PatternPropertiesValue attribute.
    properties: A PropertiesValue attribute.
    ref: A string attribute.
    required: A string attribute.
    schema: A string attribute.
    title: A string attribute.
    type: A string attribute.
    uniqueItems: A boolean attribute.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DefinitionsValue(_messages.Message):
        """A DefinitionsValue object.

    Messages:
      AdditionalProperty: An additional property for a DefinitionsValue
        object.

    Fields:
      additionalProperties: Additional properties of type DefinitionsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DefinitionsValue object.

      Fields:
        key: Name of the additional property.
        value: A JSONSchemaProps attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('JSONSchemaProps', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DependenciesValue(_messages.Message):
        """A DependenciesValue object.

    Messages:
      AdditionalProperty: An additional property for a DependenciesValue
        object.

    Fields:
      additionalProperties: Additional properties of type DependenciesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DependenciesValue object.

      Fields:
        key: Name of the additional property.
        value: A JSONSchemaPropsOrStringArray attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('JSONSchemaPropsOrStringArray', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PatternPropertiesValue(_messages.Message):
        """A PatternPropertiesValue object.

    Messages:
      AdditionalProperty: An additional property for a PatternPropertiesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        PatternPropertiesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PatternPropertiesValue object.

      Fields:
        key: Name of the additional property.
        value: A JSONSchemaProps attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('JSONSchemaProps', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PropertiesValue(_messages.Message):
        """A PropertiesValue object.

    Messages:
      AdditionalProperty: An additional property for a PropertiesValue object.

    Fields:
      additionalProperties: Additional properties of type PropertiesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PropertiesValue object.

      Fields:
        key: Name of the additional property.
        value: A JSONSchemaProps attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('JSONSchemaProps', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    additionalItems = _messages.MessageField('JSONSchemaPropsOrBool', 1)
    additionalProperties = _messages.MessageField('JSONSchemaPropsOrBool', 2)
    allOf = _messages.MessageField('JSONSchemaProps', 3, repeated=True)
    anyOf = _messages.MessageField('JSONSchemaProps', 4, repeated=True)
    default = _messages.MessageField('JSON', 5)
    definitions = _messages.MessageField('DefinitionsValue', 6)
    dependencies = _messages.MessageField('DependenciesValue', 7)
    description = _messages.StringField(8)
    enum = _messages.StringField(9, repeated=True)
    example = _messages.MessageField('JSON', 10)
    exclusiveMaximum = _messages.BooleanField(11)
    exclusiveMinimum = _messages.BooleanField(12)
    externalDocs = _messages.MessageField('ExternalDocumentation', 13)
    format = _messages.StringField(14)
    id = _messages.StringField(15)
    items = _messages.MessageField('JSONSchemaPropsOrArray', 16)
    maxItems = _messages.IntegerField(17)
    maxLength = _messages.IntegerField(18)
    maxProperties = _messages.IntegerField(19)
    maximum = _messages.FloatField(20)
    minItems = _messages.IntegerField(21)
    minLength = _messages.IntegerField(22)
    minProperties = _messages.IntegerField(23)
    minimum = _messages.FloatField(24)
    multipleOf = _messages.FloatField(25)
    not_ = _messages.MessageField('JSONSchemaProps', 26)
    oneOf = _messages.MessageField('JSONSchemaProps', 27, repeated=True)
    pattern = _messages.StringField(28)
    patternProperties = _messages.MessageField('PatternPropertiesValue', 29)
    properties = _messages.MessageField('PropertiesValue', 30)
    ref = _messages.StringField(31)
    required = _messages.StringField(32, repeated=True)
    schema = _messages.StringField(33)
    title = _messages.StringField(34)
    type = _messages.StringField(35)
    uniqueItems = _messages.BooleanField(36)