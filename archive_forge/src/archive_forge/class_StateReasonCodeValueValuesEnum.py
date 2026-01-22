from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StateReasonCodeValueValuesEnum(_messages.Enum):
    """Output only. The reason that a spoke is inactive.

    Values:
      CODE_UNSPECIFIED: No information available.
      PENDING_REVIEW: The proposed spoke is pending review.
      REJECTED: The proposed spoke has been rejected by the hub administrator.
      PAUSED: The spoke has been deactivated internally.
      FAILED: Network Connectivity Center encountered errors while accepting
        the spoke.
    """
    CODE_UNSPECIFIED = 0
    PENDING_REVIEW = 1
    REJECTED = 2
    PAUSED = 3
    FAILED = 4