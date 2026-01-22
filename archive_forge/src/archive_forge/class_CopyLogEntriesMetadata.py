from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CopyLogEntriesMetadata(_messages.Message):
    """Metadata for CopyLogEntries long running operations.

  Enums:
    StateValueValuesEnum: Output only. State of an operation.

  Fields:
    cancellationRequested: Identifies whether the user has requested
      cancellation of the operation.
    destination: Destination to which to copy log entries.For example, a Cloud
      Storage bucket:"storage.googleapis.com/my-cloud-storage-bucket"
    endTime: The end time of an operation.
    progress: Estimated progress of the operation (0 - 100%).
    request: CopyLogEntries RPC request. This field is deprecated and not
      used.
    source: Source from which to copy log entries.For example, a log
      bucket:"projects/my-project/locations/global/buckets/my-source-bucket"
    startTime: The create time of an operation.
    state: Output only. State of an operation.
    verb: Name of the verb executed by the operation.For example,"copy"
    writerIdentity: The IAM identity of a service account that must be granted
      access to the destination.If the service account is not granted
      permission to the destination within an hour, the operation will be
      cancelled.For example: "serviceAccount:foo@bar.com"
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of an operation.

    Values:
      OPERATION_STATE_UNSPECIFIED: Should not be used.
      OPERATION_STATE_SCHEDULED: The operation is scheduled.
      OPERATION_STATE_WAITING_FOR_PERMISSIONS: Waiting for necessary
        permissions.
      OPERATION_STATE_RUNNING: The operation is running.
      OPERATION_STATE_SUCCEEDED: The operation was completed successfully.
      OPERATION_STATE_FAILED: The operation failed.
      OPERATION_STATE_CANCELLED: The operation was cancelled by the user.
      OPERATION_STATE_PENDING: The operation is waiting for quota.
    """
        OPERATION_STATE_UNSPECIFIED = 0
        OPERATION_STATE_SCHEDULED = 1
        OPERATION_STATE_WAITING_FOR_PERMISSIONS = 2
        OPERATION_STATE_RUNNING = 3
        OPERATION_STATE_SUCCEEDED = 4
        OPERATION_STATE_FAILED = 5
        OPERATION_STATE_CANCELLED = 6
        OPERATION_STATE_PENDING = 7
    cancellationRequested = _messages.BooleanField(1)
    destination = _messages.StringField(2)
    endTime = _messages.StringField(3)
    progress = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    request = _messages.MessageField('CopyLogEntriesRequest', 5)
    source = _messages.StringField(6)
    startTime = _messages.StringField(7)
    state = _messages.EnumField('StateValueValuesEnum', 8)
    verb = _messages.StringField(9)
    writerIdentity = _messages.StringField(10)