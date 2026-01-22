from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OperationStage(_messages.Message):
    """Information about a particular stage of an operation.

  Enums:
    StageValueValuesEnum: The high-level stage of the operation.
    StateValueValuesEnum: Output only. State of the stage.

  Fields:
    endTime: Time the stage ended.
    metrics: Progress metric bundle.
    stage: The high-level stage of the operation.
    startTime: Time the stage started.
    state: Output only. State of the stage.
  """

    class StageValueValuesEnum(_messages.Enum):
        """The high-level stage of the operation.

    Values:
      STAGE_UNSPECIFIED: Not set.
      PREFLIGHT_CHECK: Preflight checks are running.
      CONFIGURE: Resource is being configured.
      DEPLOY: Resource is being deployed.
      HEALTH_CHECK: Waiting for the resource to become healthy.
      UPDATE: Resource is being updated.
    """
        STAGE_UNSPECIFIED = 0
        PREFLIGHT_CHECK = 1
        CONFIGURE = 2
        DEPLOY = 3
        HEALTH_CHECK = 4
        UPDATE = 5

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the stage.

    Values:
      STATE_UNSPECIFIED: Not set.
      PENDING: The stage is pending.
      RUNNING: The stage is running
      SUCCEEDED: The stage has completed successfully.
      FAILED: The stage has failed.
    """
        STATE_UNSPECIFIED = 0
        PENDING = 1
        RUNNING = 2
        SUCCEEDED = 3
        FAILED = 4
    endTime = _messages.StringField(1)
    metrics = _messages.MessageField('Metric', 2, repeated=True)
    stage = _messages.EnumField('StageValueValuesEnum', 3)
    startTime = _messages.StringField(4)
    state = _messages.EnumField('StateValueValuesEnum', 5)