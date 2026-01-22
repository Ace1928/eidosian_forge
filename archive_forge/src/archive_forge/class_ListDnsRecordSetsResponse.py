from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListDnsRecordSetsResponse(_messages.Message):
    """Represents all DNS RecordSets associated with the producer network

  Fields:
    dnsRecordSets: DNS record Set Resource
  """
    dnsRecordSets = _messages.MessageField('DnsRecordSet', 1, repeated=True)