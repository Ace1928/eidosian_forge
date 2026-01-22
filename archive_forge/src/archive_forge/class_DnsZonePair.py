from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DnsZonePair(_messages.Message):
    """* Represents a pair of private and peering DNS zone resources. *

  Fields:
    consumerPeeringZone: The DNS peering zone in the consumer project.
    producerPrivateZone: The private DNS zone in the shared producer host
      project.
  """
    consumerPeeringZone = _messages.MessageField('DnsZone', 1)
    producerPrivateZone = _messages.MessageField('DnsZone', 2)