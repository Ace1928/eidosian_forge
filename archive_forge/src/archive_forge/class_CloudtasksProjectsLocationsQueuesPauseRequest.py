from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudtasksProjectsLocationsQueuesPauseRequest(_messages.Message):
    """A CloudtasksProjectsLocationsQueuesPauseRequest object.

  Fields:
    name: Required. The queue name. For example:
      `projects/PROJECT_ID/location/LOCATION_ID/queues/QUEUE_ID`
    pauseQueueRequest: A PauseQueueRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    pauseQueueRequest = _messages.MessageField('PauseQueueRequest', 2)