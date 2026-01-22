from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1SchemaPartitionField(_messages.Message):
    """Represents a key field within the entity's partition structure. You
  could have up to 20 partition fields, but only the first 10 partitions have
  the filtering ability due to performance consideration. Note: Partition
  fields are immutable.

  Enums:
    TypeValueValuesEnum: Required. Immutable. The type of field.

  Fields:
    name: Required. Partition field name must consist of letters, numbers, and
      underscores only, with a maximum of length of 256 characters, and must
      begin with a letter or underscore..
    type: Required. Immutable. The type of field.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Required. Immutable. The type of field.

    Values:
      TYPE_UNSPECIFIED: SchemaType unspecified.
      BOOLEAN: Boolean field.
      BYTE: Single byte numeric field.
      INT16: 16-bit numeric field.
      INT32: 32-bit numeric field.
      INT64: 64-bit numeric field.
      FLOAT: Floating point numeric field.
      DOUBLE: Double precision numeric field.
      DECIMAL: Real value numeric field.
      STRING: Sequence of characters field.
      BINARY: Sequence of bytes field.
      TIMESTAMP: Date and time field.
      DATE: Date field.
      TIME: Time field.
      RECORD: Structured field. Nested fields that define the structure of the
        map. If all nested fields are nullable, this field represents a union.
      NULL: Null field that does not have values.
    """
        TYPE_UNSPECIFIED = 0
        BOOLEAN = 1
        BYTE = 2
        INT16 = 3
        INT32 = 4
        INT64 = 5
        FLOAT = 6
        DOUBLE = 7
        DECIMAL = 8
        STRING = 9
        BINARY = 10
        TIMESTAMP = 11
        DATE = 12
        TIME = 13
        RECORD = 14
        NULL = 15
    name = _messages.StringField(1)
    type = _messages.EnumField('TypeValueValuesEnum', 2)