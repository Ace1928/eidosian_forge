from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsPipelineJobsDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsPipelineJobsDeleteRequest object.

  Fields:
    name: Required. The name of the PipelineJob resource to be deleted.
      Format:
      `projects/{project}/locations/{location}/pipelineJobs/{pipeline_job}`
  """
    name = _messages.StringField(1, required=True)