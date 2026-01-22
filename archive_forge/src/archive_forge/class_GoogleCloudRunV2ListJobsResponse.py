from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRunV2ListJobsResponse(_messages.Message):
    """Response message containing a list of Jobs.

  Fields:
    jobs: The resulting list of Jobs.
    nextPageToken: A token indicating there are more items than page_size. Use
      it in the next ListJobs request to continue.
  """
    jobs = _messages.MessageField('GoogleCloudRunV2Job', 1, repeated=True)
    nextPageToken = _messages.StringField(2)