from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LockStateValueValuesEnum(_messages.Enum):
    """Output only. Current lock state of the deployment.

    Values:
      LOCK_STATE_UNSPECIFIED: The default value. This value is used if the
        lock state is omitted.
      LOCKED: The deployment is locked.
      UNLOCKED: The deployment is unlocked.
      LOCKING: The deployment is being locked.
      UNLOCKING: The deployment is being unlocked.
      LOCK_FAILED: The deployment has failed to lock.
      UNLOCK_FAILED: The deployment has failed to unlock.
    """
    LOCK_STATE_UNSPECIFIED = 0
    LOCKED = 1
    UNLOCKED = 2
    LOCKING = 3
    UNLOCKING = 4
    LOCK_FAILED = 5
    UNLOCK_FAILED = 6