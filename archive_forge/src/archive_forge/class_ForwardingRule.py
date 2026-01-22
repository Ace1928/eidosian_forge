from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ForwardingRule(_messages.Message):
    """A forwarding rule is a mapping of a `domain` to `name_servers`. This
  mapping allows VMware Engine to resolve domains for attached private clouds
  by forwarding DNS requests for a given domain to the specified nameservers.

  Fields:
    domain: Required. Domain used to resolve a `name_servers` list.
    nameServers: Required. List of DNS servers to use for domain resolution
  """
    domain = _messages.StringField(1)
    nameServers = _messages.StringField(2, repeated=True)