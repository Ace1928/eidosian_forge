from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1beta2IndexOperationMetadata(_messages.Message):
    """Metadata for google.longrunning.Operation results from
  FirestoreAdmin.CreateIndex.

  Enums:
    StateValueValuesEnum: The state of the operation.

  Fields:
    endTime: The time this operation completed. Will be unset if operation
      still in progress.
    index: The index resource that this operation is acting on. For example: `
      projects/{project_id}/databases/{database_id}/collectionGroups/{collecti
      on_id}/indexes/{index_id}`
    progressBytes: The progress, in bytes, of this operation.
    progressDocuments: The progress, in documents, of this operation.
    startTime: The time this operation started.
    state: The state of the operation.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The state of the operation.

    Values:
      OPERATION_STATE_UNSPECIFIED: Unspecified.
      INITIALIZING: Request is being prepared for processing.
      PROCESSING: Request is actively being processed.
      CANCELLING: Request is in the process of being cancelled after user
        called google.longrunning.Operations.CancelOperation on the operation.
      FINALIZING: Request has been processed and is in its finalization stage.
      SUCCESSFUL: Request has completed successfully.
      FAILED: Request has finished being processed, but encountered an error.
      CANCELLED: Request has finished being cancelled after user called
        google.longrunning.Operations.CancelOperation.
    """
        OPERATION_STATE_UNSPECIFIED = 0
        INITIALIZING = 1
        PROCESSING = 2
        CANCELLING = 3
        FINALIZING = 4
        SUCCESSFUL = 5
        FAILED = 6
        CANCELLED = 7
    endTime = _messages.StringField(1)
    index = _messages.StringField(2)
    progressBytes = _messages.MessageField('GoogleFirestoreAdminV1beta2Progress', 3)
    progressDocuments = _messages.MessageField('GoogleFirestoreAdminV1beta2Progress', 4)
    startTime = _messages.StringField(5)
    state = _messages.EnumField('StateValueValuesEnum', 6)