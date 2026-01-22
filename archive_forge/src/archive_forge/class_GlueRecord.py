from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GlueRecord(_messages.Message):
    """Defines a host on your domain that is a DNS name server for your domain
  and/or other domains. Glue records are a way of making the IP address of a
  name server known, even when it serves DNS queries for its parent domain.
  For example, when `ns.example.com` is a name server for `example.com`, the
  host `ns.example.com` must have a glue record to break the circular DNS
  reference.

  Fields:
    hostName: Required. Domain name of the host in Punycode format.
    ipv4Addresses: List of IPv4 addresses corresponding to this host in the
      standard decimal format (e.g. `198.51.100.1`). At least one of
      `ipv4_address` and `ipv6_address` must be set.
    ipv6Addresses: List of IPv6 addresses corresponding to this host in the
      standard hexadecimal format (e.g. `2001:db8::`). At least one of
      `ipv4_address` and `ipv6_address` must be set.
  """
    hostName = _messages.StringField(1)
    ipv4Addresses = _messages.StringField(2, repeated=True)
    ipv6Addresses = _messages.StringField(3, repeated=True)