from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkflowNode(_messages.Message):
    """The workflow node.

  Enums:
    StateValueValuesEnum: Output only. The node state.

  Fields:
    error: Output only. The error detail.
    jobId: Output only. The job id; populated after the node enters RUNNING
      state.
    prerequisiteStepIds: Output only. Node's prerequisite nodes.
    state: Output only. The node state.
    stepId: Output only. The name of the node.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The node state.

    Values:
      NODE_STATE_UNSPECIFIED: State is unspecified.
      BLOCKED: The node is awaiting prerequisite node to finish.
      RUNNABLE: The node is runnable but not running.
      RUNNING: The node is running.
      COMPLETED: The node completed successfully.
      FAILED: The node failed. A node can be marked FAILED because its
        ancestor or peer failed.
    """
        NODE_STATE_UNSPECIFIED = 0
        BLOCKED = 1
        RUNNABLE = 2
        RUNNING = 3
        COMPLETED = 4
        FAILED = 5
    error = _messages.StringField(1)
    jobId = _messages.StringField(2)
    prerequisiteStepIds = _messages.StringField(3, repeated=True)
    state = _messages.EnumField('StateValueValuesEnum', 4)
    stepId = _messages.StringField(5)