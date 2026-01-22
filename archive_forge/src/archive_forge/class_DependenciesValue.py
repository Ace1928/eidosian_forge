from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
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