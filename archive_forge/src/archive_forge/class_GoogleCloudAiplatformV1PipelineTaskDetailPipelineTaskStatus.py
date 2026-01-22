from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1PipelineTaskDetailPipelineTaskStatus(_messages.Message):
    """A single record of the task status.

  Enums:
    StateValueValuesEnum: Output only. The state of the task.

  Fields:
    error: Output only. The error that occurred during the state. May be set
      when the state is any of the non-final state
      (PENDING/RUNNING/CANCELLING) or FAILED state. If the state is FAILED,
      the error here is final and not going to be retried. If the state is a
      non-final state, the error indicates a system-error being retried.
    state: Output only. The state of the task.
    updateTime: Output only. Update time of this status.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the task.

    Values:
      STATE_UNSPECIFIED: Unspecified.
      PENDING: Specifies pending state for the task.
      RUNNING: Specifies task is being executed.
      SUCCEEDED: Specifies task completed successfully.
      CANCEL_PENDING: Specifies Task cancel is in pending state.
      CANCELLING: Specifies task is being cancelled.
      CANCELLED: Specifies task was cancelled.
      FAILED: Specifies task failed.
      SKIPPED: Specifies task was skipped due to cache hit.
      NOT_TRIGGERED: Specifies that the task was not triggered because the
        task's trigger policy is not satisfied. The trigger policy is
        specified in the `condition` field of PipelineJob.pipeline_spec.
    """
        STATE_UNSPECIFIED = 0
        PENDING = 1
        RUNNING = 2
        SUCCEEDED = 3
        CANCEL_PENDING = 4
        CANCELLING = 5
        CANCELLED = 6
        FAILED = 7
        SKIPPED = 8
        NOT_TRIGGERED = 9
    error = _messages.MessageField('GoogleRpcStatus', 1)
    state = _messages.EnumField('StateValueValuesEnum', 2)
    updateTime = _messages.StringField(3)