from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerWorkloadStatus(_messages.Message):
    """Workload status.

  Enums:
    StateValueValuesEnum: Output only. Workload state.

  Fields:
    detailedStatusMessage: Output only. Detailed message of the status.
    state: Output only. Workload state.
    statusMessage: Output only. Text to provide more descriptive status.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Workload state.

    Values:
      COMPOSER_WORKLOAD_STATE_UNSPECIFIED: Not able to determine the status of
        the workload.
      PENDING: Workload is in pending state and has not yet started.
      OK: Workload is running fine.
      WARNING: Workload is running but there are some non-critical problems.
      ERROR: Workload is not running due to an error.
      SUCCEEDED: Workload has finished execution with success.
      FAILED: Workload has finished execution with failure.
    """
        COMPOSER_WORKLOAD_STATE_UNSPECIFIED = 0
        PENDING = 1
        OK = 2
        WARNING = 3
        ERROR = 4
        SUCCEEDED = 5
        FAILED = 6
    detailedStatusMessage = _messages.StringField(1)
    state = _messages.EnumField('StateValueValuesEnum', 2)
    statusMessage = _messages.StringField(3)