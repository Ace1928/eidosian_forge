from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExecutionReasonValueValuesEnum(_messages.Enum):
    """Output only. A reason for the execution condition.

    Values:
      EXECUTION_REASON_UNDEFINED: Default value.
      JOB_STATUS_SERVICE_POLLING_ERROR: Internal system error getting
        execution status. System will retry.
      NON_ZERO_EXIT_CODE: A task reached its retry limit and the last attempt
        failed due to the user container exiting with a non-zero exit code.
      CANCELLED: The execution was cancelled by users.
      CANCELLING: The execution is in the process of being cancelled.
      DELETED: The execution was deleted.
    """
    EXECUTION_REASON_UNDEFINED = 0
    JOB_STATUS_SERVICE_POLLING_ERROR = 1
    NON_ZERO_EXIT_CODE = 2
    CANCELLED = 3
    CANCELLING = 4
    DELETED = 5