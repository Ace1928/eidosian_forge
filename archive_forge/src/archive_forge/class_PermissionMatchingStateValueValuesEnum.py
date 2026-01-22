from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PermissionMatchingStateValueValuesEnum(_messages.Enum):
    """Indicates whether the permission in the request is denied by the deny
    rule.

    Values:
      PERMISSION_PATTERN_MATCHING_STATE_UNSPECIFIED: Not specified.
      PERMISSION_PATTERN_MATCHED: The permission in the request matches the
        permission in the policy.
      PERMISSION_PATTERN_NOT_MATCHED: The permission in the request matches
        the permission in the policy.
    """
    PERMISSION_PATTERN_MATCHING_STATE_UNSPECIFIED = 0
    PERMISSION_PATTERN_MATCHED = 1
    PERMISSION_PATTERN_NOT_MATCHED = 2