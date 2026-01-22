from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ListDataLabelingJobsResponse(_messages.Message):
    """Response message for JobService.ListDataLabelingJobs.

  Fields:
    dataLabelingJobs: A list of DataLabelingJobs that matches the specified
      filter in the request.
    nextPageToken: The standard List next-page token.
  """
    dataLabelingJobs = _messages.MessageField('GoogleCloudAiplatformV1beta1DataLabelingJob', 1, repeated=True)
    nextPageToken = _messages.StringField(2)