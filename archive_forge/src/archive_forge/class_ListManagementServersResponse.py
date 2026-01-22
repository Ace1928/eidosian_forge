from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListManagementServersResponse(_messages.Message):
    """Response message for listing management servers.

  Fields:
    managementServers: The list of ManagementServer instances in the project
      for the specified location. If the `{location}` value in the request is
      "-", the response contains a list of instances from all locations. In
      case any location is unreachable, the response will only return
      management servers in reachable locations and the 'unreachable' field
      will be populated with a list of unreachable locations.
    nextPageToken: A token identifying a page of results the server should
      return.
    unreachable: Locations that could not be reached.
  """
    managementServers = _messages.MessageField('ManagementServer', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)