from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatabaseEncryption(_messages.Message):
    """Configuration of etcd encryption.

  Enums:
    CurrentStateValueValuesEnum: Output only. The current state of etcd
      encryption.
    StateValueValuesEnum: The desired state of etcd encryption.

  Fields:
    currentState: Output only. The current state of etcd encryption.
    decryptionKeys: Output only. Keys in use by the cluster for decrypting
      existing objects, in addition to the key in `key_name`. Each item is a
      CloudKMS key resource.
    keyName: Name of CloudKMS key to use for the encryption of secrets in
      etcd. Ex. projects/my-project/locations/global/keyRings/my-
      ring/cryptoKeys/my-key
    lastOperationErrors: Output only. Records errors seen during
      DatabaseEncryption update operations.
    state: The desired state of etcd encryption.
  """

    class CurrentStateValueValuesEnum(_messages.Enum):
        """Output only. The current state of etcd encryption.

    Values:
      CURRENT_STATE_UNSPECIFIED: Should never be set
      CURRENT_STATE_ENCRYPTED: Secrets in etcd are encrypted.
      CURRENT_STATE_DECRYPTED: Secrets in etcd are stored in plain text (at
        etcd level) - this is unrelated to Compute Engine level full disk
        encryption.
      CURRENT_STATE_ENCRYPTION_PENDING: Encryption (or re-encryption with a
        different CloudKMS key) of Secrets is in progress.
      CURRENT_STATE_ENCRYPTION_ERROR: Encryption (or re-encryption with a
        different CloudKMS key) of Secrets in etcd encountered an error.
      CURRENT_STATE_DECRYPTION_PENDING: De-crypting Secrets to plain text in
        etcd is in progress.
      CURRENT_STATE_DECRYPTION_ERROR: De-crypting Secrets to plain text in
        etcd encountered an error.
    """
        CURRENT_STATE_UNSPECIFIED = 0
        CURRENT_STATE_ENCRYPTED = 1
        CURRENT_STATE_DECRYPTED = 2
        CURRENT_STATE_ENCRYPTION_PENDING = 3
        CURRENT_STATE_ENCRYPTION_ERROR = 4
        CURRENT_STATE_DECRYPTION_PENDING = 5
        CURRENT_STATE_DECRYPTION_ERROR = 6

    class StateValueValuesEnum(_messages.Enum):
        """The desired state of etcd encryption.

    Values:
      UNKNOWN: Should never be set
      ENCRYPTED: Secrets in etcd are encrypted.
      DECRYPTED: Secrets in etcd are stored in plain text (at etcd level) -
        this is unrelated to Compute Engine level full disk encryption.
    """
        UNKNOWN = 0
        ENCRYPTED = 1
        DECRYPTED = 2
    currentState = _messages.EnumField('CurrentStateValueValuesEnum', 1)
    decryptionKeys = _messages.StringField(2, repeated=True)
    keyName = _messages.StringField(3)
    lastOperationErrors = _messages.MessageField('OperationError', 4, repeated=True)
    state = _messages.EnumField('StateValueValuesEnum', 5)