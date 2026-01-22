from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StoragetransferTransferOperationsResumeRequest(_messages.Message):
    """A StoragetransferTransferOperationsResumeRequest object.

  Fields:
    name: Required. The name of the transfer operation.
    resumeTransferOperationRequest: A ResumeTransferOperationRequest resource
      to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    resumeTransferOperationRequest = _messages.MessageField('ResumeTransferOperationRequest', 2)