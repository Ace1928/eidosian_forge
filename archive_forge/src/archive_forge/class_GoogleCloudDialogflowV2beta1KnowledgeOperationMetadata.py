from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1KnowledgeOperationMetadata(_messages.Message):
    """Metadata in google::longrunning::Operation for Knowledge operations.

  Enums:
    StateValueValuesEnum: Required. Output only. The current state of this
      operation.

  Fields:
    exportOperationMetadata: Metadata for the Export Data Operation such as
      the destination of export.
    knowledgeBase: The name of the knowledge base interacted with during the
      operation.
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
    exportOperationMetadata = _messages.MessageField('GoogleCloudDialogflowV2beta1ExportOperationMetadata', 1)
    knowledgeBase = _messages.StringField(2)
    state = _messages.EnumField('StateValueValuesEnum', 3)