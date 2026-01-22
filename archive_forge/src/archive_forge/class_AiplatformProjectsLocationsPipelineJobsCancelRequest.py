from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsPipelineJobsCancelRequest(_messages.Message):
    """A AiplatformProjectsLocationsPipelineJobsCancelRequest object.

  Fields:
    googleCloudAiplatformV1CancelPipelineJobRequest: A
      GoogleCloudAiplatformV1CancelPipelineJobRequest resource to be passed as
      the request body.
    name: Required. The name of the PipelineJob to cancel. Format:
      `projects/{project}/locations/{location}/pipelineJobs/{pipeline_job}`
  """
    googleCloudAiplatformV1CancelPipelineJobRequest = _messages.MessageField('GoogleCloudAiplatformV1CancelPipelineJobRequest', 1)
    name = _messages.StringField(2, required=True)