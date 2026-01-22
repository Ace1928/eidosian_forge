from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsJobsDebugSendCaptureRequest(_messages.Message):
    """A DataflowProjectsJobsDebugSendCaptureRequest object.

  Fields:
    jobId: The job id.
    projectId: The project id.
    sendDebugCaptureRequest: A SendDebugCaptureRequest resource to be passed
      as the request body.
  """
    jobId = _messages.StringField(1, required=True)
    projectId = _messages.StringField(2, required=True)
    sendDebugCaptureRequest = _messages.MessageField('SendDebugCaptureRequest', 3)