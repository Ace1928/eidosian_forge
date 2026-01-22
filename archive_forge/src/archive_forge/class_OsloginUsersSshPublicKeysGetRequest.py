from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsloginUsersSshPublicKeysGetRequest(_messages.Message):
    """A OsloginUsersSshPublicKeysGetRequest object.

  Fields:
    name: Required. The fingerprint of the public key to retrieve. Public keys
      are identified by their SHA-256 fingerprint. The fingerprint of the
      public key is in format `users/{user}/sshPublicKeys/{fingerprint}`.
  """
    name = _messages.StringField(1, required=True)