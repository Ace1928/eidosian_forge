from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudtasksProjectsLocationsQueuesTasksLeaseRequest(_messages.Message):
    """A CloudtasksProjectsLocationsQueuesTasksLeaseRequest object.

  Fields:
    leaseTasksRequest: A LeaseTasksRequest resource to be passed as the
      request body.
    parent: Required. The queue name. For example:
      `projects/PROJECT_ID/locations/LOCATION_ID/queues/QUEUE_ID`
  """
    leaseTasksRequest = _messages.MessageField('LeaseTasksRequest', 1)
    parent = _messages.StringField(2, required=True)