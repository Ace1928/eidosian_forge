from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SignSshPublicKeyRequest(_messages.Message):
    """A SignSshPublicKeyRequest object.

  Fields:
    sshPublicKey: The SSH public key to sign.
  """
    sshPublicKey = _messages.StringField(1)