from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EdgecontainerProjectsLocationsVpnConnectionsCreateRequest(_messages.Message):
    """A EdgecontainerProjectsLocationsVpnConnectionsCreateRequest object.

  Fields:
    parent: Required. The parent location where this vpn connection will be
      created.
    requestId: A unique identifier for this request. Restricted to 36 ASCII
      characters. A random UUID is recommended. This request is only
      idempotent if `request_id` is provided.
    vpnConnection: A VpnConnection resource to be passed as the request body.
    vpnConnectionId: Required. The VPN connection identifier.
  """
    parent = _messages.StringField(1, required=True)
    requestId = _messages.StringField(2)
    vpnConnection = _messages.MessageField('VpnConnection', 3)
    vpnConnectionId = _messages.StringField(4)