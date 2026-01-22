from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateInstanceMetadataItemsRequest(_messages.Message):
    """Request for adding/changing metadata items for an instance.

  Messages:
    ItemsValue: Metadata items to add/update for the instance.

  Fields:
    items: Metadata items to add/update for the instance.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ItemsValue(_messages.Message):
        """Metadata items to add/update for the instance.

    Messages:
      AdditionalProperty: An additional property for a ItemsValue object.

    Fields:
      additionalProperties: Additional properties of type ItemsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ItemsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    items = _messages.MessageField('ItemsValue', 1)