from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AddDnsZoneRequest(_messages.Message):
    """Request to add a private managed DNS zone in the shared producer host
  project and a matching DNS peering zone in the consumer project.

  Fields:
    consumerNetwork: Required. The network that the consumer is using to
      connect with services. Must be in the form of
      projects/{project}/global/networks/{network} {project} is the project
      number, as in '12345' {network} is the network name.
    dnsSuffix: Required. The DNS name suffix for the zones e.g.
      `example.com.`. Cloud DNS requires that a DNS suffix ends with a
      trailing dot.
    name: Required. The name for both the private zone in the shared producer
      host project and the peering zone in the consumer project. Must be
      unique within both projects. The name must be 1-63 characters long, must
      begin with a letter, end with a letter or digit, and only contain
      lowercase letters, digits or dashes.
  """
    consumerNetwork = _messages.StringField(1)
    dnsSuffix = _messages.StringField(2)
    name = _messages.StringField(3)