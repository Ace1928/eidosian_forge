from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StoragetransferTransferJobsPatchRequest(_messages.Message):
    """A StoragetransferTransferJobsPatchRequest object.

  Fields:
    jobName: Required. The name of job to update.
    updateTransferJobRequest: A UpdateTransferJobRequest resource to be passed
      as the request body.
  """
    jobName = _messages.StringField(1, required=True)
    updateTransferJobRequest = _messages.MessageField('UpdateTransferJobRequest', 2)