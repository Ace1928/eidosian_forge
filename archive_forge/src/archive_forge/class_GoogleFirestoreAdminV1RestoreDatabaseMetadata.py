from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1RestoreDatabaseMetadata(_messages.Message):
    """Metadata for the long-running operation from the RestoreDatabase
  request.

  Enums:
    OperationStateValueValuesEnum: The operation state of the restore.

  Fields:
    backup: The name of the backup restoring from.
    database: The name of the database being restored to.
    endTime: The time the restore finished, unset for ongoing restores.
    operationState: The operation state of the restore.
    progressPercentage: How far along the restore is as an estimated
      percentage of remaining time.
    startTime: The time the restore was started.
  """

    class OperationStateValueValuesEnum(_messages.Enum):
        """The operation state of the restore.

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
    backup = _messages.StringField(1)
    database = _messages.StringField(2)
    endTime = _messages.StringField(3)
    operationState = _messages.EnumField('OperationStateValueValuesEnum', 4)
    progressPercentage = _messages.MessageField('GoogleFirestoreAdminV1Progress', 5)
    startTime = _messages.StringField(6)