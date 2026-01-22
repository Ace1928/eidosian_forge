from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RemoveDnsZoneRequest(_messages.Message):
    """Request to remove a private managed DNS zone in the shared producer host
  project and a matching DNS peering zone in the consumer project.

  Fields:
    consumerNetwork: Required. The network that the consumer is using to
      connect with services. Must be in the form of
      projects/{project}/global/networks/{network} {project} is the project
      number, as in '12345' {network} is the network name.
    name: Required. The name for both the private zone in the shared producer
      host project and the peering zone in the consumer project.
  """
    consumerNetwork = _messages.StringField(1)
    name = _messages.StringField(2)