from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PostKeyRevocationActionTypeValueValuesEnum(_messages.Enum):
    """PostKeyRevocationActionType of the instance.

    Values:
      NOOP: Indicates user chose no operation.
      POST_KEY_REVOCATION_ACTION_TYPE_UNSPECIFIED: Default value. This value
        is unused.
      SHUTDOWN: Indicates user chose to opt for VM shutdown on key revocation.
    """
    NOOP = 0
    POST_KEY_REVOCATION_ACTION_TYPE_UNSPECIFIED = 1
    SHUTDOWN = 2