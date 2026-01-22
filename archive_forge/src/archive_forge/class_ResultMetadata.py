from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResultMetadata(_messages.Message):
    """Metadata of result field.

  Enums:
    DataTypeValueValuesEnum: The data type of the field.

  Fields:
    dataType: The data type of the field.
    description: A brief description of the field.
    field: Name of the result field.
  """

    class DataTypeValueValuesEnum(_messages.Enum):
        """The data type of the field.

    Values:
      DATA_TYPE_UNSPECIFIED: Data type is not specified.
      DATA_TYPE_INT: DEPRECATED! Use DATA_TYPE_INTEGER.
      DATA_TYPE_SMALLINT: Short integer(int16) data type.
      DATA_TYPE_DOUBLE: Double data type.
      DATA_TYPE_DATE: Date data type.
      DATA_TYPE_DATETIME: DEPRECATED! Use DATA_TYPE_TIMESTAMP.
      DATA_TYPE_TIME: Time data type.
      DATA_TYPE_STRING: DEPRECATED! Use DATA_TYPE_VARCHAR.
      DATA_TYPE_LONG: DEPRECATED! Use DATA_TYPE_BIGINT.
      DATA_TYPE_BOOLEAN: Boolean data type.
      DATA_TYPE_DECIMAL: Decimal data type.
      DATA_TYPE_UUID: DEPRECATED! Use DATA_TYPE_VARCHAR.
      DATA_TYPE_BLOB: UNSUPPORTED! Binary data type.
      DATA_TYPE_BIT: Bit data type.
      DATA_TYPE_TINYINT: Small integer(int8) data type.
      DATA_TYPE_INTEGER: Integer(int32) data type.
      DATA_TYPE_BIGINT: Long integer(int64) data type.
      DATA_TYPE_FLOAT: Float data type.
      DATA_TYPE_REAL: Real data type.
      DATA_TYPE_NUMERIC: Numeric data type.
      DATA_TYPE_CHAR: Char data type.
      DATA_TYPE_VARCHAR: Varchar data type.
      DATA_TYPE_LONGVARCHAR: Longvarchar data type.
      DATA_TYPE_TIMESTAMP: Timestamp data type.
      DATA_TYPE_NCHAR: Nchar data type.
      DATA_TYPE_NVARCHAR: Nvarchar data type.
      DATA_TYPE_LONGNVARCHAR: Longnvarchar data type.
      DATA_TYPE_NULL: Null data type.
      DATA_TYPE_OTHER: UNSUPPORTED! Binary data type.
      DATA_TYPE_JAVA_OBJECT: UNSUPPORTED! Binary data type.
      DATA_TYPE_DISTINCT: UNSUPPORTED! Binary data type.
      DATA_TYPE_STRUCT: UNSUPPORTED! Binary data type.
      DATA_TYPE_ARRAY: UNSUPPORTED! Binary data type.
      DATA_TYPE_CLOB: UNSUPPORTED! Binary data type.
      DATA_TYPE_REF: UNSUPPORTED! Binary data type.
      DATA_TYPE_DATALINK: UNSUPPORTED! Binary data type.
      DATA_TYPE_ROWID: UNSUPPORTED! Row id data type.
      DATA_TYPE_BINARY: UNSUPPORTED! Binary data type.
      DATA_TYPE_VARBINARY: UNSUPPORTED! Variable binary data type.
      DATA_TYPE_LONGVARBINARY: UNSUPPORTED! Long variable binary data type.
      DATA_TYPE_NCLOB: UNSUPPORTED! NCLOB data type.
      DATA_TYPE_SQLXML: UNSUPPORTED! SQL XML data type is not supported.
      DATA_TYPE_REF_CURSOR: UNSUPPORTED! Cursor reference type is not
        supported.
      DATA_TYPE_TIME_WITH_TIMEZONE: UNSUPPORTED! Use TIME or TIMESTAMP
        instead.
      DATA_TYPE_TIMESTAMP_WITH_TIMEZONE: UNSUPPORTED! Use TIMESTAMP instead.
    """
        DATA_TYPE_UNSPECIFIED = 0
        DATA_TYPE_INT = 1
        DATA_TYPE_SMALLINT = 2
        DATA_TYPE_DOUBLE = 3
        DATA_TYPE_DATE = 4
        DATA_TYPE_DATETIME = 5
        DATA_TYPE_TIME = 6
        DATA_TYPE_STRING = 7
        DATA_TYPE_LONG = 8
        DATA_TYPE_BOOLEAN = 9
        DATA_TYPE_DECIMAL = 10
        DATA_TYPE_UUID = 11
        DATA_TYPE_BLOB = 12
        DATA_TYPE_BIT = 13
        DATA_TYPE_TINYINT = 14
        DATA_TYPE_INTEGER = 15
        DATA_TYPE_BIGINT = 16
        DATA_TYPE_FLOAT = 17
        DATA_TYPE_REAL = 18
        DATA_TYPE_NUMERIC = 19
        DATA_TYPE_CHAR = 20
        DATA_TYPE_VARCHAR = 21
        DATA_TYPE_LONGVARCHAR = 22
        DATA_TYPE_TIMESTAMP = 23
        DATA_TYPE_NCHAR = 24
        DATA_TYPE_NVARCHAR = 25
        DATA_TYPE_LONGNVARCHAR = 26
        DATA_TYPE_NULL = 27
        DATA_TYPE_OTHER = 28
        DATA_TYPE_JAVA_OBJECT = 29
        DATA_TYPE_DISTINCT = 30
        DATA_TYPE_STRUCT = 31
        DATA_TYPE_ARRAY = 32
        DATA_TYPE_CLOB = 33
        DATA_TYPE_REF = 34
        DATA_TYPE_DATALINK = 35
        DATA_TYPE_ROWID = 36
        DATA_TYPE_BINARY = 37
        DATA_TYPE_VARBINARY = 38
        DATA_TYPE_LONGVARBINARY = 39
        DATA_TYPE_NCLOB = 40
        DATA_TYPE_SQLXML = 41
        DATA_TYPE_REF_CURSOR = 42
        DATA_TYPE_TIME_WITH_TIMEZONE = 43
        DATA_TYPE_TIMESTAMP_WITH_TIMEZONE = 44
    dataType = _messages.EnumField('DataTypeValueValuesEnum', 1)
    description = _messages.StringField(2)
    field = _messages.StringField(3)