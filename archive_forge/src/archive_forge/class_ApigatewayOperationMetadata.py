from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigatewayOperationMetadata(_messages.Message):
    """Represents the metadata of the long-running operation.

  Fields:
    apiVersion: Output only. API version used to start the operation.
    createTime: Output only. The time the operation was created.
    diagnostics: Output only. Diagnostics generated during processing of
      configuration source files.
    endTime: Output only. The time the operation finished running.
    requestedCancellation: Output only. Identifies whether the user has
      requested cancellation of the operation. Operations that have
      successfully been cancelled have Operation.error value with a
      google.rpc.Status.code of 1, corresponding to `Code.CANCELLED`.
    statusMessage: Output only. Human-readable status of the operation, if
      any.
    target: Output only. Server-defined resource path for the target of the
      operation.
    verb: Output only. Name of the verb executed by the operation.
  """
    apiVersion = _messages.StringField(1)
    createTime = _messages.StringField(2)
    diagnostics = _messages.MessageField('ApigatewayOperationMetadataDiagnostic', 3, repeated=True)
    endTime = _messages.StringField(4)
    requestedCancellation = _messages.BooleanField(5)
    statusMessage = _messages.StringField(6)
    target = _messages.StringField(7)
    verb = _messages.StringField(8)