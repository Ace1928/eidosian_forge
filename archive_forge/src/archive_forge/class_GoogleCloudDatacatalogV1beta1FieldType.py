from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1FieldType(_messages.Message):
    """A GoogleCloudDatacatalogV1beta1FieldType object.

  Enums:
    PrimitiveTypeValueValuesEnum: Represents primitive types - string, bool
      etc.

  Fields:
    enumType: Represents an enum type.
    primitiveType: Represents primitive types - string, bool etc.
  """

    class PrimitiveTypeValueValuesEnum(_messages.Enum):
        """Represents primitive types - string, bool etc.

    Values:
      PRIMITIVE_TYPE_UNSPECIFIED: This is the default invalid value for a
        type.
      DOUBLE: A double precision number.
      STRING: An UTF-8 string.
      BOOL: A boolean value.
      TIMESTAMP: A timestamp.
    """
        PRIMITIVE_TYPE_UNSPECIFIED = 0
        DOUBLE = 1
        STRING = 2
        BOOL = 3
        TIMESTAMP = 4
    enumType = _messages.MessageField('GoogleCloudDatacatalogV1beta1FieldTypeEnumType', 1)
    primitiveType = _messages.EnumField('PrimitiveTypeValueValuesEnum', 2)