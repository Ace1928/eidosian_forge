from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DnsPeering(_messages.Message):
    """DNS peering configuration. These configurations are used to create DNS
  peering with the customer Cloud DNS.

  Fields:
    description: Optional. Optional description of the dns zone.
    domain: Required. The dns name suffix of the zone.
    name: Required. The resource name of the dns peering zone. Format: project
      s/{project}/locations/{location}/instances/{instance}/dnsPeerings/{dns_p
      eering}
    targetNetwork: Optional. Optional target network to which dns peering
      should happen.
    targetProject: Optional. Optional target project to which dns peering
      should happen.
  """
    description = _messages.StringField(1)
    domain = _messages.StringField(2)
    name = _messages.StringField(3)
    targetNetwork = _messages.StringField(4)
    targetProject = _messages.StringField(5)