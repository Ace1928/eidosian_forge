from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventTypesValueListEntryValuesEnum(_messages.Enum):
    """EventTypesValueListEntryValuesEnum enum type.

    Values:
      EVENT_TYPE_UNSPECIFIED: Illegal value, to avoid allowing a default.
      TRANSFER_OPERATION_SUCCESS: `TransferOperation` completed with status
        SUCCESS.
      TRANSFER_OPERATION_FAILED: `TransferOperation` completed with status
        FAILED.
      TRANSFER_OPERATION_ABORTED: `TransferOperation` completed with status
        ABORTED.
    """
    EVENT_TYPE_UNSPECIFIED = 0
    TRANSFER_OPERATION_SUCCESS = 1
    TRANSFER_OPERATION_FAILED = 2
    TRANSFER_OPERATION_ABORTED = 3