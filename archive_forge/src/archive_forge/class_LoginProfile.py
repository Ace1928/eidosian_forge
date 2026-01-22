from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class LoginProfile(_messages.Message):
    """The user profile information used for logging in to a virtual machine on
  Google Compute Engine.

  Messages:
    SshPublicKeysValue: A map from SSH public key fingerprint to the
      associated key object.

  Fields:
    name: Required. A unique user ID.
    posixAccounts: The list of POSIX accounts associated with the user.
    securityKeys: The registered security key credentials for a user.
    sshPublicKeys: A map from SSH public key fingerprint to the associated key
      object.
  """

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
    name = _messages.StringField(1)
    posixAccounts = _messages.MessageField('PosixAccount', 2, repeated=True)
    securityKeys = _messages.MessageField('SecurityKey', 3, repeated=True)
    sshPublicKeys = _messages.MessageField('SshPublicKeysValue', 4)