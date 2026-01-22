from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class DetectedHotKeysValue(_messages.Message):
    """Debugging information for each detected hot key. Keyed by a hash of
    the key.

    Messages:
      AdditionalProperty: An additional property for a DetectedHotKeysValue
        object.

    Fields:
      additionalProperties: Additional properties of type DetectedHotKeysValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a DetectedHotKeysValue object.

      Fields:
        key: Name of the additional property.
        value: A HotKeyInfo attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('HotKeyInfo', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)