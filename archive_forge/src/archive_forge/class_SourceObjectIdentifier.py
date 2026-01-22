from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourceObjectIdentifier(_messages.Message):
    """Represents an identifier of an object in the data source.

  Fields:
    mysqlIdentifier: Mysql data source object identifier.
    oracleIdentifier: Oracle data source object identifier.
    postgresqlIdentifier: PostgreSQL data source object identifier.
    sqlServerIdentifier: SQLServer data source object identifier.
  """
    mysqlIdentifier = _messages.MessageField('MysqlObjectIdentifier', 1)
    oracleIdentifier = _messages.MessageField('OracleObjectIdentifier', 2)
    postgresqlIdentifier = _messages.MessageField('PostgresqlObjectIdentifier', 3)
    sqlServerIdentifier = _messages.MessageField('SqlServerObjectIdentifier', 4)