from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class AssetTypeConfigsValue(_messages.Message):
    """A map between asset type name and its configuration within this
    catalog.

    Messages:
      AdditionalProperty: An additional property for a AssetTypeConfigsValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        AssetTypeConfigsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a AssetTypeConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A AssetTypeConfig attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('AssetTypeConfig', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)