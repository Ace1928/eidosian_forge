from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1ExportDocumentsMetadata(_messages.Message):
    """Metadata for google.longrunning.Operation results from
  FirestoreAdmin.ExportDocuments.

  Enums:
    OperationStateValueValuesEnum: The state of the export operation.

  Fields:
    collectionIds: Which collection ids are being exported.
    endTime: The time this operation completed. Will be unset if operation
      still in progress.
    namespaceIds: Which namespace ids are being exported.
    operationState: The state of the export operation.
    outputUriPrefix: Where the documents are being exported to.
    progressBytes: The progress, in bytes, of this operation.
    progressDocuments: The progress, in documents, of this operation.
    snapshotTime: The timestamp that corresponds to the version of the
      database that is being exported. If unspecified, there are no guarantees
      about the consistency of the documents being exported.
    startTime: The time this operation started.
  """

    class OperationStateValueValuesEnum(_messages.Enum):
        """The state of the export operation.

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
    collectionIds = _messages.StringField(1, repeated=True)
    endTime = _messages.StringField(2)
    namespaceIds = _messages.StringField(3, repeated=True)
    operationState = _messages.EnumField('OperationStateValueValuesEnum', 4)
    outputUriPrefix = _messages.StringField(5)
    progressBytes = _messages.MessageField('GoogleFirestoreAdminV1Progress', 6)
    progressDocuments = _messages.MessageField('GoogleFirestoreAdminV1Progress', 7)
    snapshotTime = _messages.StringField(8)
    startTime = _messages.StringField(9)