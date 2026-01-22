from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateBackupMetadata(_messages.Message):
    """Metadata type for the operation returned by CreateBackup.

  Fields:
    cancelTime: The time at which cancellation of this operation was received.
      Operations.CancelOperation starts asynchronous cancellation on a long-
      running operation. The server makes a best effort to cancel the
      operation, but success is not guaranteed. Clients can use
      Operations.GetOperation or other methods to check whether the
      cancellation succeeded or whether the operation completed despite
      cancellation. On successful cancellation, the operation is not deleted;
      instead, it becomes an operation with an Operation.error value with a
      google.rpc.Status.code of 1, corresponding to `Code.CANCELLED`.
    database: The name of the database the backup is created from.
    name: The name of the backup being created.
    progress: The progress of the CreateBackup operation.
  """
    cancelTime = _messages.StringField(1)
    database = _messages.StringField(2)
    name = _messages.StringField(3)
    progress = _messages.MessageField('OperationProgress', 4)