from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StoragetransferTransferJobsRunRequest(_messages.Message):
    """A StoragetransferTransferJobsRunRequest object.

  Fields:
    jobName: Required. The name of the transfer job.
    runTransferJobRequest: A RunTransferJobRequest resource to be passed as
      the request body.
  """
    jobName = _messages.StringField(1, required=True)
    runTransferJobRequest = _messages.MessageField('RunTransferJobRequest', 2)