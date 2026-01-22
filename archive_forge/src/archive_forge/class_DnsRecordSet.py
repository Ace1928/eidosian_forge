from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DnsRecordSet(_messages.Message):
    """Represents a DNS record set resource.

  Fields:
    data: Required. As defined in RFC 1035 (section 5) and RFC 1034 (section
      3.6.1) for examples see https://cloud.google.com/dns/records/json-
      record.
    domain: Required. The DNS or domain name of the record set, e.g.
      `test.example.com`. Cloud DNS requires that a DNS suffix ends with a
      trailing dot.
    ttl: Required. The period of time for which this RecordSet can be cached
      by resolvers.
    type: Required. The identifier of a supported record type.
  """
    data = _messages.StringField(1, repeated=True)
    domain = _messages.StringField(2)
    ttl = _messages.StringField(3)
    type = _messages.StringField(4)