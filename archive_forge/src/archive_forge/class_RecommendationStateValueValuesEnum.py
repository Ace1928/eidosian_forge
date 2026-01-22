from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecommendationStateValueValuesEnum(_messages.Enum):
    """Required. Recommendation state

    Values:
      UNSPECIFIED: <no description>
      ACTIVE: Recommendation is active and can be applied. ACTIVE
        recommendations can be marked as CLAIMED, SUCCEEDED, or FAILED.
      CLAIMED: Recommendation is in claimed state. Recommendations content is
        immutable and cannot be updated by Google. CLAIMED recommendations can
        be marked as CLAIMED, SUCCEEDED, or FAILED.
      SUCCEEDED: Recommendation is in succeeded state. Recommendations content
        is immutable and cannot be updated by Google. SUCCEEDED
        recommendations can be marked as SUCCEEDED, or FAILED.
      FAILED: Recommendation is in failed state. Recommendations content is
        immutable and cannot be updated by Google. FAILED recommendations can
        be marked as SUCCEEDED, or FAILED.
      DISMISSED: Recommendation is in dismissed state. Recommendation content
        can be updated by Google. DISMISSED recommendations can be marked as
        ACTIVE.
    """
    UNSPECIFIED = 0
    ACTIVE = 1
    CLAIMED = 2
    SUCCEEDED = 3
    FAILED = 4
    DISMISSED = 5