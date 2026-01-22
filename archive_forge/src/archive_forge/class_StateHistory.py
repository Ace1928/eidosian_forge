from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StateHistory(_messages.Message):
    """Historical state information.

  Enums:
    StateValueValuesEnum: Output only. The state of the batch at this point in
      history.

  Fields:
    state: Output only. The state of the batch at this point in history.
    stateMessage: Output only. Details about the state at this point in
      history.
    stateStartTime: Output only. The time when the batch entered the
      historical state.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the batch at this point in history.

    Values:
      STATE_UNSPECIFIED: The batch state is unknown.
      PENDING: The batch is created before running.
      RUNNING: The batch is running.
      CANCELLING: The batch is cancelling.
      CANCELLED: The batch cancellation was successful.
      SUCCEEDED: The batch completed successfully.
      FAILED: The batch is no longer running due to an error.
    """
        STATE_UNSPECIFIED = 0
        PENDING = 1
        RUNNING = 2
        CANCELLING = 3
        CANCELLED = 4
        SUCCEEDED = 5
        FAILED = 6
    state = _messages.EnumField('StateValueValuesEnum', 1)
    stateMessage = _messages.StringField(2)
    stateStartTime = _messages.StringField(3)