from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class KeyRangeInfosValue(_messages.Message):
    """A (sparse) mapping from key bucket index to the KeyRangeInfos for that
    key bucket.

    Messages:
      AdditionalProperty: An additional property for a KeyRangeInfosValue
        object.

    Fields:
      additionalProperties: Additional properties of type KeyRangeInfosValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a KeyRangeInfosValue object.

      Fields:
        key: Name of the additional property.
        value: A KeyRangeInfos attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('KeyRangeInfos', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)