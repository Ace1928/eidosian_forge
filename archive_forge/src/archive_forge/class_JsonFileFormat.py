from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JsonFileFormat(_messages.Message):
    """JSON file format configuration.

  Enums:
    CompressionValueValuesEnum: Compression of the loaded JSON file.
    SchemaFileFormatValueValuesEnum: The schema file format along JSON data
      files.

  Fields:
    compression: Compression of the loaded JSON file.
    schemaFileFormat: The schema file format along JSON data files.
  """

    class CompressionValueValuesEnum(_messages.Enum):
        """Compression of the loaded JSON file.

    Values:
      JSON_COMPRESSION_UNSPECIFIED: Unspecified json file compression.
      NO_COMPRESSION: Do not compress JSON file.
      GZIP: Gzip compression.
    """
        JSON_COMPRESSION_UNSPECIFIED = 0
        NO_COMPRESSION = 1
        GZIP = 2

    class SchemaFileFormatValueValuesEnum(_messages.Enum):
        """The schema file format along JSON data files.

    Values:
      SCHEMA_FILE_FORMAT_UNSPECIFIED: Unspecified schema file format.
      NO_SCHEMA_FILE: Do not attach schema file.
      AVRO_SCHEMA_FILE: Avro schema format.
    """
        SCHEMA_FILE_FORMAT_UNSPECIFIED = 0
        NO_SCHEMA_FILE = 1
        AVRO_SCHEMA_FILE = 2
    compression = _messages.EnumField('CompressionValueValuesEnum', 1)
    schemaFileFormat = _messages.EnumField('SchemaFileFormatValueValuesEnum', 2)