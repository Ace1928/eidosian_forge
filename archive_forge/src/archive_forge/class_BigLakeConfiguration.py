from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigLakeConfiguration(_messages.Message):
    """Configuration for BigLake managed tables.

  Enums:
    FileFormatValueValuesEnum: Required. The file format the table data is
      stored in.
    TableFormatValueValuesEnum: Required. The table format the metadata only
      snapshots are stored in.

  Fields:
    connectionId: Required. The connection specifying the credentials to be
      used to read and write to external storage, such as Cloud Storage. The
      connection_id can have the form
      "<project\\_id>.<location\\_id>.<connection\\_id>" or "projects/<project\\_i
      d>/locations/<location\\_id>/connections/<connection\\_id>".
    fileFormat: Required. The file format the table data is stored in.
    storageUri: Required. The fully qualified location prefix of the external
      folder where table data is stored. The '*' wildcard character is not
      allowed. The URI should be in the format "gs://bucket/path_to_table/"
    tableFormat: Required. The table format the metadata only snapshots are
      stored in.
  """

    class FileFormatValueValuesEnum(_messages.Enum):
        """Required. The file format the table data is stored in.

    Values:
      FILE_FORMAT_UNSPECIFIED: Default Value.
      PARQUET: Apache Parquet format.
    """
        FILE_FORMAT_UNSPECIFIED = 0
        PARQUET = 1

    class TableFormatValueValuesEnum(_messages.Enum):
        """Required. The table format the metadata only snapshots are stored in.

    Values:
      TABLE_FORMAT_UNSPECIFIED: Default Value.
      ICEBERG: Apache Iceberg format.
    """
        TABLE_FORMAT_UNSPECIFIED = 0
        ICEBERG = 1
    connectionId = _messages.StringField(1)
    fileFormat = _messages.EnumField('FileFormatValueValuesEnum', 2)
    storageUri = _messages.StringField(3)
    tableFormat = _messages.EnumField('TableFormatValueValuesEnum', 4)