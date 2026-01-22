from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CallLogLevelValueValuesEnum(_messages.Enum):
    """The call logging level associated to this execution.

    Values:
      CALL_LOG_LEVEL_UNSPECIFIED: No call logging level specified.
      LOG_ALL_CALLS: Log all call steps within workflows, all call returns,
        and all exceptions raised.
      LOG_ERRORS_ONLY: Log only exceptions that are raised from call steps
        within workflows.
    """
    CALL_LOG_LEVEL_UNSPECIFIED = 0
    LOG_ALL_CALLS = 1
    LOG_ERRORS_ONLY = 2