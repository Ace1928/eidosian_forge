from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransferLockStateValueValuesEnum(_messages.Enum):
    """Indicates whether the domain is protected by a transfer lock. For a
    transfer to succeed, this must show `UNLOCKED`. To unlock a domain, go to
    its current registrar.

    Values:
      TRANSFER_LOCK_STATE_UNSPECIFIED: The state is unspecified.
      UNLOCKED: The domain is unlocked and can be transferred to another
        registrar.
      LOCKED: The domain is locked and cannot be transferred to another
        registrar.
    """
    TRANSFER_LOCK_STATE_UNSPECIFIED = 0
    UNLOCKED = 1
    LOCKED = 2