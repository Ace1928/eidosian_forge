from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LinkedVpnTunnels(_messages.Message):
    """A collection of Cloud VPN tunnel resources. These resources should be
  redundant HA VPN tunnels that all advertise the same prefixes to Google
  Cloud. Alternatively, in a passive/active configuration, all tunnels should
  be capable of advertising the same prefixes.

  Fields:
    siteToSiteDataTransfer: A value that controls whether site-to-site data
      transfer is enabled for these resources. Data transfer is available only
      in [supported locations](https://cloud.google.com/network-
      connectivity/docs/network-connectivity-center/concepts/locations).
    uris: The URIs of linked VPN tunnel resources.
    vpcNetwork: Output only. The VPC network where these VPN tunnels are
      located.
  """
    siteToSiteDataTransfer = _messages.BooleanField(1)
    uris = _messages.StringField(2, repeated=True)
    vpcNetwork = _messages.StringField(3)