from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsPipelineJobsCreateRequest(_messages.Message):
    """A AiplatformProjectsLocationsPipelineJobsCreateRequest object.

  Fields:
    googleCloudAiplatformV1PipelineJob: A GoogleCloudAiplatformV1PipelineJob
      resource to be passed as the request body.
    parent: Required. The resource name of the Location to create the
      PipelineJob in. Format: `projects/{project}/locations/{location}`
    pipelineJobId: The ID to use for the PipelineJob, which will become the
      final component of the PipelineJob name. If not provided, an ID will be
      automatically generated. This value should be less than 128 characters,
      and valid characters are `/a-z-/`.
  """
    googleCloudAiplatformV1PipelineJob = _messages.MessageField('GoogleCloudAiplatformV1PipelineJob', 1)
    parent = _messages.StringField(2, required=True)
    pipelineJobId = _messages.StringField(3)