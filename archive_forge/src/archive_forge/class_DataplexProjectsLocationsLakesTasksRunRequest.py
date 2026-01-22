from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesTasksRunRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesTasksRunRequest object.

  Fields:
    googleCloudDataplexV1RunTaskRequest: A GoogleCloudDataplexV1RunTaskRequest
      resource to be passed as the request body.
    name: Required. The resource name of the task: projects/{project_number}/l
      ocations/{location_id}/lakes/{lake_id}/tasks/{task_id}.
  """
    googleCloudDataplexV1RunTaskRequest = _messages.MessageField('GoogleCloudDataplexV1RunTaskRequest', 1)
    name = _messages.StringField(2, required=True)