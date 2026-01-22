from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GrpcRouteHeaderMatch(_messages.Message):
    """A match against a collection of headers.

  Enums:
    TypeValueValuesEnum: Optional. Specifies how to match against the value of
      the header. If not specified, a default value of EXACT is used.

  Fields:
    key: Required. The key of the header.
    type: Optional. Specifies how to match against the value of the header. If
      not specified, a default value of EXACT is used.
    value: Required. The value of the header.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Optional. Specifies how to match against the value of the header. If
    not specified, a default value of EXACT is used.

    Values:
      TYPE_UNSPECIFIED: Unspecified.
      EXACT: Will only match the exact value provided.
      REGULAR_EXPRESSION: Will match paths conforming to the prefix specified
        by value. RE2 syntax is supported.
    """
        TYPE_UNSPECIFIED = 0
        EXACT = 1
        REGULAR_EXPRESSION = 2
    key = _messages.StringField(1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)
    value = _messages.StringField(3)