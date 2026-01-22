from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RolloutInfo(_messages.Message):
    """RolloutInfo represents the state of package deployment at all the
  clusters the rollout is targeting.

  Enums:
    StateValueValuesEnum: Output only. state conveys the overall status of the
      Rollout.

  Fields:
    clusterInfo: resource bundle's deployment status for each target cluster.
    message: Output only. Message convey additional information related to the
      rollout.
    reconciliationEndTime: Output only. Timestamp when reconciliation ends for
      the overall rollout.
    reconciliationStartTime: Output only. Timestamp when reconciliation starts
      for the overall rollout.
    state: Output only. state conveys the overall status of the Rollout.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. state conveys the overall status of the Rollout.

    Values:
      STATE_UNSPECIFIED: Unspecified state.
      COMPLETED: Rollout completed.
      SUSPENDED: Rollout suspended.
      ABORTED: Rollout aborted.
      ERRORED: Rollout errored.
      IN_PROGRESS: Rollout in progress.
      STALLED: Rollout stalled.
    """
        STATE_UNSPECIFIED = 0
        COMPLETED = 1
        SUSPENDED = 2
        ABORTED = 3
        ERRORED = 4
        IN_PROGRESS = 5
        STALLED = 6
    clusterInfo = _messages.MessageField('ClusterInfo', 1, repeated=True)
    message = _messages.StringField(2)
    reconciliationEndTime = _messages.StringField(3)
    reconciliationStartTime = _messages.StringField(4)
    state = _messages.EnumField('StateValueValuesEnum', 5)