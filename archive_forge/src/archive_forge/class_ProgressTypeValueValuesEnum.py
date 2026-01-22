from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ProgressTypeValueValuesEnum(_messages.Enum):
    """Progress type of this step entry.

    Values:
      PROGRESS_TYPE_UNSPECIFIED: Current step entry does not have any progress
        data.
      PROGRESS_TYPE_FOR: Current step entry is in progress of a FOR step.
      PROGRESS_TYPE_SWITCH: Current step entry is in progress of a SWITCH
        step.
      PROGRESS_TYPE_RETRY: Current step entry is in progress of a RETRY step.
      PROGRESS_TYPE_PARALLEL_FOR: Current step entry is in progress of a
        PARALLEL FOR step.
      PROGRESS_TYPE_PARALLEL_BRANCH: Current step entry is in progress of a
        PARALLEL BRANCH step.
    """
    PROGRESS_TYPE_UNSPECIFIED = 0
    PROGRESS_TYPE_FOR = 1
    PROGRESS_TYPE_SWITCH = 2
    PROGRESS_TYPE_RETRY = 3
    PROGRESS_TYPE_PARALLEL_FOR = 4
    PROGRESS_TYPE_PARALLEL_BRANCH = 5