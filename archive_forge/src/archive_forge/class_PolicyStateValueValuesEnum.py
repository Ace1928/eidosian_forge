from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PolicyStateValueValuesEnum(_messages.Enum):
    """Indicates if a policy tag has been applied to the column.

    Values:
      COLUMN_POLICY_STATE_UNSPECIFIED: No policy tags.
      COLUMN_POLICY_TAGGED: Column has policy tag applied.
    """
    COLUMN_POLICY_STATE_UNSPECIFIED = 0
    COLUMN_POLICY_TAGGED = 1