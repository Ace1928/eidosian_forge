from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FaultApplyStatus(_messages.Message):
    """Message describing Faults and its apply status apphub targets.

  Enums:
    StateValueValuesEnum: Output only. Message describing fault status

  Fields:
    fault: Message describing the fault config
    faultTarget: Message describing the fault target.
    state: Output only. Message describing fault status
    status: Output only. Message describing status code/error when calling PO
    targetStatuses: List of target statuses.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Message describing fault status

    Values:
      STATE_UNSPECIFIED: Default state.
      APPLY_IN_PROGRESS: Apply fault is queued.
      APPLIED: Fault apply is completed.
      STOP_IN_PROGRESS: Stop apply in progress
      STOPPED: fault apply stopped
      ABORTED: Fault apply aborted.
      APPLY_ERRORED: Fault apply errored.
      STOP_ERRORED: Fault stop errored.
    """
        STATE_UNSPECIFIED = 0
        APPLY_IN_PROGRESS = 1
        APPLIED = 2
        STOP_IN_PROGRESS = 3
        STOPPED = 4
        ABORTED = 5
        APPLY_ERRORED = 6
        STOP_ERRORED = 7
    fault = _messages.MessageField('Fault', 1)
    faultTarget = _messages.MessageField('FaultInjectionTarget', 2)
    state = _messages.EnumField('StateValueValuesEnum', 3)
    status = _messages.MessageField('Status', 4)
    targetStatuses = _messages.MessageField('TargetStatus', 5, repeated=True)