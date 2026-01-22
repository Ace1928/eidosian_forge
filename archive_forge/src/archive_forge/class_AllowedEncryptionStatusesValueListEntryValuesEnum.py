from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllowedEncryptionStatusesValueListEntryValuesEnum(_messages.Enum):
    """AllowedEncryptionStatusesValueListEntryValuesEnum enum type.

    Values:
      ENCRYPTION_UNSPECIFIED: The encryption status of the device is not
        specified or not known.
      ENCRYPTION_UNSUPPORTED: The device does not support encryption.
      UNENCRYPTED: The device supports encryption, but is currently
        unencrypted.
      ENCRYPTED: The device is encrypted.
    """
    ENCRYPTION_UNSPECIFIED = 0
    ENCRYPTION_UNSUPPORTED = 1
    UNENCRYPTED = 2
    ENCRYPTED = 3