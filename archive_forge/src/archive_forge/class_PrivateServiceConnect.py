from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivateServiceConnect(_messages.Message):
    """Defines the Private Service Connect network configuration for the pool.

  Fields:
    networkAttachment: Required. Immutable. The network attachment that the
      worker network interface is peered to. Must be in the format `projects/{
      project}/regions/{region}/networkAttachments/{networkAttachment}`. The
      region of network attachment must be the same as the worker pool. See
      [Network Attachments](https://cloud.google.com/vpc/docs/about-network-
      attachments)
    publicIpAddressDisabled: Required. Immutable. Disable public IP on the
      primary network interface. If true, workers are created without any
      public address, which prevents network egress to public IPs unless a
      network proxy is configured. If false, workers are created with a public
      address which allows for public internet egress. The public address only
      applies to traffic through the primary network interface. If
      `route_all_traffic` is set to true, all traffic will go through the non-
      primary network interface, this boolean has no effect.
    routeAllTraffic: Immutable. Route all traffic through PSC interface.
      Enable this if you want full control of traffic in the private pool.
      Configure Cloud NAT for the subnet of network attachment if you need to
      access public Internet. If false, Only route private IPs, e.g.
      10.0.0.0/8, 172.16.0.0/12, and 192.168.0.0/16 through PSC interface.
  """
    networkAttachment = _messages.StringField(1)
    publicIpAddressDisabled = _messages.BooleanField(2)
    routeAllTraffic = _messages.BooleanField(3)