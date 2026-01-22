from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
class ResourceRecordSet(_messages.Message):
    """A unit of data that will be returned by the DNS servers.

  Fields:
    kind: Identifies what kind of resource this is. Value: the fixed string
      "dns#resourceRecordSet".
    name: For example, www.example.com.
    rrdatas: As defined in RFC 1035 (section 5) and RFC 1034 (section 3.6.1).
    ttl: Number of seconds that this ResourceRecordSet can be cached by
      resolvers.
    type: The identifier of a supported record type, for example, A, AAAA, MX,
      TXT, and so on.
  """
    kind = _messages.StringField(1, default=u'dns#resourceRecordSet')
    name = _messages.StringField(2)
    rrdatas = _messages.StringField(3, repeated=True)
    ttl = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    type = _messages.StringField(5)