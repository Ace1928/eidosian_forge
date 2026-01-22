from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudtasksProjectsLocationsQueuesPatchRequest(_messages.Message):
    """A CloudtasksProjectsLocationsQueuesPatchRequest object.

  Fields:
    name: Caller-specified and required in CreateQueue, after which it becomes
      output only. The queue name. The queue name must have the following
      format: `projects/PROJECT_ID/locations/LOCATION_ID/queues/QUEUE_ID` *
      `PROJECT_ID` can contain letters ([A-Za-z]), numbers ([0-9]), hyphens
      (-), colons (:), or periods (.). For more information, see [Identifying
      projects](https://cloud.google.com/resource-manager/docs/creating-
      managing-projects#identifying_projects) * `LOCATION_ID` is the canonical
      ID for the queue's location. The list of available locations can be
      obtained by calling ListLocations. For more information, see
      https://cloud.google.com/about/locations/. * `QUEUE_ID` can contain
      letters ([A-Za-z]), numbers ([0-9]), or hyphens (-). The maximum length
      is 100 characters.
    queue: A Queue resource to be passed as the request body.
    updateMask: A mask used to specify which fields of the queue are being
      updated. If empty, then all fields will be updated.
  """
    name = _messages.StringField(1, required=True)
    queue = _messages.MessageField('Queue', 2)
    updateMask = _messages.StringField(3)