from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EncryptionKeyRevocationActionValueValuesEnum(_messages.Enum):
    """The action to take if the encryption key is revoked.

    Values:
      ENCRYPTION_KEY_REVOCATION_ACTION_UNSPECIFIED: Unspecified
      PREVENT_NEW: Prevents the creation of new instances.
      SHUTDOWN: Shuts down existing instances, and prevents creation of new
        ones.
    """
    ENCRYPTION_KEY_REVOCATION_ACTION_UNSPECIFIED = 0
    PREVENT_NEW = 1
    SHUTDOWN = 2