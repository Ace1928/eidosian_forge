from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpApplicationsV1alphaApplicationOperationMetadata(_messages.Message):
    """Represents the metadata of the long-running operation.

  Fields:
    createTime: Output only. The time the operation was created.
    endTime: Output only. The time the operation finished running.
    requestedCancellation: Output only. Identifies whether the user has
      requested cancellation of the operation. Operations that have been
      cancelled successfully have Operation.error value with a
      google.rpc.Status.code of 1, corresponding to `Code.CANCELLED`.
    statusMessage: Output only. Human-readable status of the operation, if
      any.
    target: Output only. Server-defined resource path for the target of the
      operation.
    verb: Output only. Name of the verb executed by the operation.
  """
    createTime = _messages.StringField(1)
    endTime = _messages.StringField(2)
    requestedCancellation = _messages.BooleanField(3)
    statusMessage = _messages.StringField(4)
    target = _messages.StringField(5)
    verb = _messages.StringField(6)