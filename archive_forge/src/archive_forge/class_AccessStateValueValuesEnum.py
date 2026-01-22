from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccessStateValueValuesEnum(_messages.Enum):
    """Whether the principal in the access tuple has permission to access the
    resource in the access tuple under the given policies.

    Values:
      ACCESS_STATE_UNSPECIFIED: Default value. This value is unused.
      GRANTED: The principal has the permission.
      NOT_GRANTED: The principal does not have the permission.
      UNKNOWN_CONDITIONAL: The principal has the permission only if a
        condition expression evaluates to `true`.
      UNKNOWN_INFO_DENIED: The user who created the Replay does not have
        access to all of the policies that Policy Simulator needs to evaluate.
    """
    ACCESS_STATE_UNSPECIFIED = 0
    GRANTED = 1
    NOT_GRANTED = 2
    UNKNOWN_CONDITIONAL = 3
    UNKNOWN_INFO_DENIED = 4