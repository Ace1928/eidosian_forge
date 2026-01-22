from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatabaseDump(_messages.Message):
    """A specification of the location of and metadata about a database dump
  from a relational database management system.

  Enums:
    DatabaseTypeValueValuesEnum: The type of the database.
    TypeValueValuesEnum: Optional. The type of the database dump. If
      unspecified, defaults to MYSQL.

  Fields:
    databaseType: The type of the database.
    gcsUri: A Cloud Storage object or folder URI that specifies the source
      from which to import metadata. It must begin with gs://.
    sourceDatabase: The name of the source database.
    type: Optional. The type of the database dump. If unspecified, defaults to
      MYSQL.
  """

    class DatabaseTypeValueValuesEnum(_messages.Enum):
        """The type of the database.

    Values:
      DATABASE_TYPE_UNSPECIFIED: The type of the source database is unknown.
      MYSQL: The type of the source database is MySQL.
    """
        DATABASE_TYPE_UNSPECIFIED = 0
        MYSQL = 1

    class TypeValueValuesEnum(_messages.Enum):
        """Optional. The type of the database dump. If unspecified, defaults to
    MYSQL.

    Values:
      TYPE_UNSPECIFIED: The type of the database dump is unknown.
      MYSQL: Database dump is a MySQL dump file.
      AVRO: Database dump contains Avro files.
    """
        TYPE_UNSPECIFIED = 0
        MYSQL = 1
        AVRO = 2
    databaseType = _messages.EnumField('DatabaseTypeValueValuesEnum', 1)
    gcsUri = _messages.StringField(2)
    sourceDatabase = _messages.StringField(3)
    type = _messages.EnumField('TypeValueValuesEnum', 4)