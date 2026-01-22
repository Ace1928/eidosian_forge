from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudtasksProjectsLocationsQueuesTasksAcknowledgeRequest(_messages.Message):
    """A CloudtasksProjectsLocationsQueuesTasksAcknowledgeRequest object.

  Fields:
    acknowledgeTaskRequest: A AcknowledgeTaskRequest resource to be passed as
      the request body.
    name: Required. The task name. For example:
      `projects/PROJECT_ID/locations/LOCATION_ID/queues/QUEUE_ID/tasks/TASK_ID
      `
  """
    acknowledgeTaskRequest = _messages.MessageField('AcknowledgeTaskRequest', 1)
    name = _messages.StringField(2, required=True)