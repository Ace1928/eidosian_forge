from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FailureReasonsValueListEntryValuesEnum(_messages.Enum):
    """FailureReasonsValueListEntryValuesEnum enum type.

    Values:
      FAILURE_REASON_UNSPECIFIED: Failure reason is not assigned.
      FAILED_INTENT: Whether NLU failed to recognize user intent.
      FAILED_WEBHOOK: Whether webhook failed during the turn.
    """
    FAILURE_REASON_UNSPECIFIED = 0
    FAILED_INTENT = 1
    FAILED_WEBHOOK = 2