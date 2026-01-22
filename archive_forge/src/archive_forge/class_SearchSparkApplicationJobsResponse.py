from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SearchSparkApplicationJobsResponse(_messages.Message):
    """A list of Jobs associated with a Spark Application.

  Fields:
    nextPageToken: This token is included in the response if there are more
      results to fetch. To fetch additional results, provide this value as the
      page_token in a subsequent SearchSparkApplicationJobsRequest.
    sparkApplicationJobs: Output only. Data corresponding to a spark job.
  """
    nextPageToken = _messages.StringField(1)
    sparkApplicationJobs = _messages.MessageField('JobData', 2, repeated=True)