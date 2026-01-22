from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateWorkerPoolOperationMetadata(_messages.Message):
    """Metadata for the `CreateWorkerPool` operation.

  Fields:
    completeTime: Time the operation was completed.
    createTime: Time the operation was created.
    workerPool: The resource name of the `WorkerPool` to create. Format:
      `projects/{project}/locations/{location}/workerPools/{worker_pool}`.
  """
    completeTime = _messages.StringField(1)
    createTime = _messages.StringField(2)
    workerPool = _messages.StringField(3)