from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemotebuildexecutionAdminV1alphaUpdateWorkerPoolRequest(_messages.Message):
    """The request used for UpdateWorkerPool.

  Fields:
    updateMask: The update mask applies to worker_pool. For the `FieldMask`
      definition, see https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#fieldmask If an empty update_mask
      is provided, only the non-default valued field in the worker pool field
      will be updated. Note that in order to update a field to the default
      value (zero, false, empty string) an explicit update_mask must be
      provided.
    workerPool: Specifies the worker pool to update.
  """
    updateMask = _messages.StringField(1)
    workerPool = _messages.MessageField('GoogleDevtoolsRemotebuildexecutionAdminV1alphaWorkerPool', 2)