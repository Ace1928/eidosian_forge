from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListConnectionsResponse(_messages.Message):
    """ListConnectionsResponse is the response to list peering states for the
  given service and consumer project.

  Fields:
    connections: The list of Connections.
  """
    connections = _messages.MessageField('Connection', 1, repeated=True)