from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GlobalNetworkEndpointGroupsDetachEndpointsRequest(_messages.Message):
    """A GlobalNetworkEndpointGroupsDetachEndpointsRequest object.

  Fields:
    networkEndpoints: The list of network endpoints to be detached.
  """
    networkEndpoints = _messages.MessageField('NetworkEndpoint', 1, repeated=True)