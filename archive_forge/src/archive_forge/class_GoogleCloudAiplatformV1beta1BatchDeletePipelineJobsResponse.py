from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1BatchDeletePipelineJobsResponse(_messages.Message):
    """Response message for PipelineService.BatchDeletePipelineJobs.

  Fields:
    pipelineJobs: PipelineJobs deleted.
  """
    pipelineJobs = _messages.MessageField('GoogleCloudAiplatformV1beta1PipelineJob', 1, repeated=True)