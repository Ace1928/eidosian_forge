from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BgpRouteAsPath(_messages.Message):
    """A BgpRouteAsPath object.

  Enums:
    TypeValueValuesEnum: [Output only] Type of AS-PATH segment (SEQUENCE or
      SET)

  Fields:
    asns: [Output only] ASNs in the path segment. When type is SEQUENCE, these
      are ordered.
    type: [Output only] Type of AS-PATH segment (SEQUENCE or SET)
  """

    class TypeValueValuesEnum(_messages.Enum):
        """[Output only] Type of AS-PATH segment (SEQUENCE or SET)

    Values:
      AS_PATH_TYPE_SEQUENCE: <no description>
      AS_PATH_TYPE_SET: <no description>
    """
        AS_PATH_TYPE_SEQUENCE = 0
        AS_PATH_TYPE_SET = 1
    asns = _messages.IntegerField(1, repeated=True, variant=_messages.Variant.INT32)
    type = _messages.EnumField('TypeValueValuesEnum', 2)