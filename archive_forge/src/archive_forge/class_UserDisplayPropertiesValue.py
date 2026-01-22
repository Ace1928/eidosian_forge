from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class UserDisplayPropertiesValue(_messages.Message):
    """List of display properties to help UI filter jobs.

    Messages:
      AdditionalProperty: An additional property for a
        UserDisplayPropertiesValue object.

    Fields:
      additionalProperties: Additional properties of type
        UserDisplayPropertiesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a UserDisplayPropertiesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)