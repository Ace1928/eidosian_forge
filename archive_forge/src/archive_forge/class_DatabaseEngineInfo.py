from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatabaseEngineInfo(_messages.Message):
    """The type and version of a source or destination database.

  Enums:
    EngineValueValuesEnum: Required. Engine type.

  Fields:
    engine: Required. Engine type.
    version: Required. Engine version, for example "12.c.1".
  """

    class EngineValueValuesEnum(_messages.Enum):
        """Required. Engine type.

    Values:
      DATABASE_ENGINE_UNSPECIFIED: The source database engine of the migration
        job is unknown.
      MYSQL: The source engine is MySQL.
      POSTGRESQL: The source engine is PostgreSQL.
      SQLSERVER: The source engine is SQL Server.
      ORACLE: The source engine is Oracle.
    """
        DATABASE_ENGINE_UNSPECIFIED = 0
        MYSQL = 1
        POSTGRESQL = 2
        SQLSERVER = 3
        ORACLE = 4
    engine = _messages.EnumField('EngineValueValuesEnum', 1)
    version = _messages.StringField(2)