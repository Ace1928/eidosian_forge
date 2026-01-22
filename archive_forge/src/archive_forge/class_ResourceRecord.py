from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceRecord(_messages.Message):
    """A DNS resource record.

  Enums:
    TypeValueValuesEnum: Resource record type. Example: `AAAA`.

  Fields:
    name: Relative name of the object affected by this record. Only applicable
      for `CNAME` records. Example: 'www'.
    rrdata: Data for this record. Values vary by record type, as defined in
      RFC 1035 (section 5) and RFC 1034 (section 3.6.1).
    type: Resource record type. Example: `AAAA`.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Resource record type. Example: `AAAA`.

    Values:
      RECORD_TYPE_UNSPECIFIED: An unknown resource record.
      A: An A resource record. Data is an IPv4 address.
      AAAA: An AAAA resource record. Data is an IPv6 address.
      CNAME: A CNAME resource record. Data is a domain name to be aliased.
    """
        RECORD_TYPE_UNSPECIFIED = 0
        A = 1
        AAAA = 2
        CNAME = 3
    name = _messages.StringField(1)
    rrdata = _messages.StringField(2)
    type = _messages.EnumField('TypeValueValuesEnum', 3)