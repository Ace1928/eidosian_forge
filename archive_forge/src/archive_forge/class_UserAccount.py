from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserAccount(_messages.Message):
    """User account provisioned for the customer.

  Fields:
    encryptedPassword: Encrypted initial password value.
    kmsKeyVersion: KMS CryptoKey Version used to encrypt the password.
  """
    encryptedPassword = _messages.StringField(1)
    kmsKeyVersion = _messages.StringField(2)