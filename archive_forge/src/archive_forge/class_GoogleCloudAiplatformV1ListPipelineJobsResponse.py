from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ListPipelineJobsResponse(_messages.Message):
    """Response message for PipelineService.ListPipelineJobs

  Fields:
    nextPageToken: A token to retrieve the next page of results. Pass to
      ListPipelineJobsRequest.page_token to obtain that page.
    pipelineJobs: List of PipelineJobs in the requested page.
  """
    nextPageToken = _messages.StringField(1)
    pipelineJobs = _messages.MessageField('GoogleCloudAiplatformV1PipelineJob', 2, repeated=True)