from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudtasksProjectsLocationsQueuesCreateRequest(_messages.Message):
    """A CloudtasksProjectsLocationsQueuesCreateRequest object.

  Fields:
    parent: Required. The location name in which the queue will be created.
      For example: `projects/PROJECT_ID/locations/LOCATION_ID` The list of
      allowed locations can be obtained by calling Cloud Tasks' implementation
      of ListLocations.
    queue: A Queue resource to be passed as the request body.
  """
    parent = _messages.StringField(1, required=True)
    queue = _messages.MessageField('Queue', 2)