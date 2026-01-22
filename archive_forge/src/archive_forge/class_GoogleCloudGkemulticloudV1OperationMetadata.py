from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1OperationMetadata(_messages.Message):
    """Metadata about a long-running operation.

  Fields:
    createTime: Output only. The time at which this operation was created.
    endTime: Output only. The time at which this operation was completed.
    errorDetail: Output only. Human-readable status of any error that occurred
      during the operation.
    requestedCancellation: Output only. Identifies whether it has been
      requested cancellation for the operation. Operations that have
      successfully been cancelled have Operation.error value with a
      google.rpc.Status.code of 1, corresponding to `Code.CANCELLED`.
    statusDetail: Output only. Human-readable status of the operation, if any.
    target: Output only. The name of the resource associated to this
      operation.
    verb: Output only. The verb associated with the API method which triggered
      this operation. Possible values are "create", "delete", "update" and
      "import".
  """
    createTime = _messages.StringField(1)
    endTime = _messages.StringField(2)
    errorDetail = _messages.StringField(3)
    requestedCancellation = _messages.BooleanField(4)
    statusDetail = _messages.StringField(5)
    target = _messages.StringField(6)
    verb = _messages.StringField(7)