from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ItemsValue(_messages.Message):
    """Inventory items related to the VM keyed by an opaque unique identifier
    for each inventory item. The identifier is unique to each distinct and
    addressable inventory item and will change, when there is a new package
    version.

    Messages:
      AdditionalProperty: An additional property for a ItemsValue object.

    Fields:
      additionalProperties: Additional properties of type ItemsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ItemsValue object.

      Fields:
        key: Name of the additional property.
        value: A Item attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('Item', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)