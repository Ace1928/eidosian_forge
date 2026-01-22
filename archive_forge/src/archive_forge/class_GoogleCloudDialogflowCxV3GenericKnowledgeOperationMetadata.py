from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3GenericKnowledgeOperationMetadata(_messages.Message):
    """Metadata in google::longrunning::Operation for Knowledge operations.

  Enums:
    StateValueValuesEnum: Required. Output only. The current state of this
      operation.

  Fields:
    state: Required. Output only. The current state of this operation.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Required. Output only. The current state of this operation.

    Values:
      STATE_UNSPECIFIED: State unspecified.
      PENDING: The operation has been created.
      RUNNING: The operation is currently running.
      DONE: The operation is done, either cancelled or completed.
    """
        STATE_UNSPECIFIED = 0
        PENDING = 1
        RUNNING = 2
        DONE = 3
    state = _messages.EnumField('StateValueValuesEnum', 1)