from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class DimensionPassedValue(_messages.Message):
    """The result of each dimension for data quality result. The key of the
    map is the name of the dimension. The value is the bool value depicting
    whether the dimension result was pass or not.

    Messages:
      AdditionalProperty: An additional property for a DimensionPassedValue
        object.

    Fields:
      additionalProperties: Additional properties of type DimensionPassedValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a DimensionPassedValue object.

      Fields:
        key: Name of the additional property.
        value: A boolean attribute.
      """
        key = _messages.StringField(1)
        value = _messages.BooleanField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)