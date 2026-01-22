from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RollbackAttempt(_messages.Message):
    """RollbackAttempt represents an action of rolling back a Cloud Deploy
  'Target'.

  Enums:
    StateValueValuesEnum: Output only. Valid state of this rollback action.

  Fields:
    destinationPhase: Output only. The phase to which the rollout will be
      rolled back to.
    rolloutId: Output only. ID of the rollback `Rollout` to create.
    state: Output only. Valid state of this rollback action.
    stateDesc: Output only. Description of the state of the Rollback.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Valid state of this rollback action.

    Values:
      REPAIR_STATE_UNSPECIFIED: The `repair` has an unspecified state.
      REPAIR_STATE_SUCCEEDED: The `repair` action has succeeded.
      REPAIR_STATE_CANCELLED: The `repair` action was cancelled.
      REPAIR_STATE_FAILED: The `repair` action has failed.
      REPAIR_STATE_IN_PROGRESS: The `repair` action is in progress.
      REPAIR_STATE_PENDING: The `repair` action is pending.
      REPAIR_STATE_SKIPPED: The `repair` action was skipped.
      REPAIR_STATE_ABORTED: The `repair` action was aborted.
    """
        REPAIR_STATE_UNSPECIFIED = 0
        REPAIR_STATE_SUCCEEDED = 1
        REPAIR_STATE_CANCELLED = 2
        REPAIR_STATE_FAILED = 3
        REPAIR_STATE_IN_PROGRESS = 4
        REPAIR_STATE_PENDING = 5
        REPAIR_STATE_SKIPPED = 6
        REPAIR_STATE_ABORTED = 7
    destinationPhase = _messages.StringField(1)
    rolloutId = _messages.StringField(2)
    state = _messages.EnumField('StateValueValuesEnum', 3)
    stateDesc = _messages.StringField(4)