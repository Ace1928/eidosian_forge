from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1ListJobsResponse(_messages.Message):
    """Response message for the ListJobs method.

  Fields:
    jobs: The list of jobs.
    nextPageToken: Optional. Pass this token as the `page_token` field of the
      request for a subsequent call.
  """
    jobs = _messages.MessageField('GoogleCloudMlV1Job', 1, repeated=True)
    nextPageToken = _messages.StringField(2)