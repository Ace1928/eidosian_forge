from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListDnsZonesResponse(_messages.Message):
    """Represents all DNS zones in the shared producer host project and the
  matching peering zones in the consumer project.

  Fields:
    dnsZonePairs: All pairs of private DNS zones in the shared producer host
      project and the matching peering zones in the consumer project..
  """
    dnsZonePairs = _messages.MessageField('DnsZonePair', 1, repeated=True)