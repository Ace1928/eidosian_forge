from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EncryptionStatusValueValuesEnum(_messages.Enum):
    """How the table is encrypted.

    Values:
      ENCRYPTION_STATUS_UNSPECIFIED: Unused.
      ENCRYPTION_GOOGLE_MANAGED: Google manages server-side encryption keys on
        your behalf.
      ENCRYPTION_CUSTOMER_MANAGED: Customer provides the key.
    """
    ENCRYPTION_STATUS_UNSPECIFIED = 0
    ENCRYPTION_GOOGLE_MANAGED = 1
    ENCRYPTION_CUSTOMER_MANAGED = 2