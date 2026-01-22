from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1ListJobsResponse(_messages.Message):
    """List jobs response.

  Fields:
    jobs: Jobs under a given task.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    jobs = _messages.MessageField('GoogleCloudDataplexV1Job', 1, repeated=True)
    nextPageToken = _messages.StringField(2)