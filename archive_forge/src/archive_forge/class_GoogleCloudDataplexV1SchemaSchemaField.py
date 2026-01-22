from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1SchemaSchemaField(_messages.Message):
    """Represents a column field within a table schema.

  Enums:
    ModeValueValuesEnum: Required. Additional field semantics.
    TypeValueValuesEnum: Required. The type of field.

  Fields:
    description: Optional. User friendly field description. Must be less than
      or equal to 1024 characters.
    fields: Optional. Any nested field for complex types.
    mode: Required. Additional field semantics.
    name: Required. The name of the field. Must contain only letters, numbers
      and underscores, with a maximum length of 767 characters, and must begin
      with a letter or underscore.
    type: Required. The type of field.
  """

    class ModeValueValuesEnum(_messages.Enum):
        """Required. Additional field semantics.

    Values:
      MODE_UNSPECIFIED: Mode unspecified.
      REQUIRED: The field has required semantics.
      NULLABLE: The field has optional semantics, and may be null.
      REPEATED: The field has repeated (0 or more) semantics, and is a list of
        values.
    """
        MODE_UNSPECIFIED = 0
        REQUIRED = 1
        NULLABLE = 2
        REPEATED = 3

    class TypeValueValuesEnum(_messages.Enum):
        """Required. The type of field.

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
    description = _messages.StringField(1)
    fields = _messages.MessageField('GoogleCloudDataplexV1SchemaSchemaField', 2, repeated=True)
    mode = _messages.EnumField('ModeValueValuesEnum', 3)
    name = _messages.StringField(4)
    type = _messages.EnumField('TypeValueValuesEnum', 5)