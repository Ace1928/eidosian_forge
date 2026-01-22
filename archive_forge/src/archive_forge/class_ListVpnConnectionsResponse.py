from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListVpnConnectionsResponse(_messages.Message):
    """List of VPN connections in a location.

  Fields:
    nextPageToken: A token to retrieve next page of results.
    unreachable: Locations that could not be reached.
    vpnConnections: VpnConnections in the location.
  """
    nextPageToken = _messages.StringField(1)
    unreachable = _messages.StringField(2, repeated=True)
    vpnConnections = _messages.MessageField('VpnConnection', 3, repeated=True)