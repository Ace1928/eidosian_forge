from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsloginUsersSshPublicKeysPatchRequest(_messages.Message):
    """A OsloginUsersSshPublicKeysPatchRequest object.

  Fields:
    name: Required. The fingerprint of the public key to update. Public keys
      are identified by their SHA-256 fingerprint. The fingerprint of the
      public key is in format `users/{user}/sshPublicKeys/{fingerprint}`.
    sshPublicKey: A SshPublicKey resource to be passed as the request body.
    updateMask: Mask to control which fields get updated. Updates all if not
      present.
  """
    name = _messages.StringField(1, required=True)
    sshPublicKey = _messages.MessageField('SshPublicKey', 2)
    updateMask = _messages.StringField(3)