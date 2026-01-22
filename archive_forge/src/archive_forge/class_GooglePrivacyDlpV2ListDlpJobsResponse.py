from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2ListDlpJobsResponse(_messages.Message):
    """The response message for listing DLP jobs.

  Fields:
    jobs: A list of DlpJobs that matches the specified filter in the request.
    nextPageToken: The standard List next-page token.
  """
    jobs = _messages.MessageField('GooglePrivacyDlpV2DlpJob', 1, repeated=True)
    nextPageToken = _messages.StringField(2)