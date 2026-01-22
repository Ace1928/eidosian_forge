from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
@encoding.MapUnrecognizedFields('additionalProperties')
class SshPublicKeysValue(_messages.Message):
    """A map from SSH public key fingerprint to the associated key object.

    Messages:
      AdditionalProperty: An additional property for a SshPublicKeysValue
        object.

    Fields:
      additionalProperties: Additional properties of type SshPublicKeysValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a SshPublicKeysValue object.

      Fields:
        key: Name of the additional property.
        value: A SshPublicKey attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('SshPublicKey', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)