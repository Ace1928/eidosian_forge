from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatapipelinesProjectsLocationsPipelinesStopRequest(_messages.Message):
    """A DatapipelinesProjectsLocationsPipelinesStopRequest object.

  Fields:
    googleCloudDatapipelinesV1StopPipelineRequest: A
      GoogleCloudDatapipelinesV1StopPipelineRequest resource to be passed as
      the request body.
    name: Required. The pipeline name. For example:
      `projects/PROJECT_ID/locations/LOCATION_ID/pipelines/PIPELINE_ID`.
  """
    googleCloudDatapipelinesV1StopPipelineRequest = _messages.MessageField('GoogleCloudDatapipelinesV1StopPipelineRequest', 1)
    name = _messages.StringField(2, required=True)