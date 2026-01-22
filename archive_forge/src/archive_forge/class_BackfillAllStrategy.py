from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackfillAllStrategy(_messages.Message):
    """Backfill strategy to automatically backfill the Stream's objects.
  Specific objects can be excluded.

  Fields:
    mysqlExcludedObjects: MySQL data source objects to avoid backfilling.
    oracleExcludedObjects: Oracle data source objects to avoid backfilling.
    postgresqlExcludedObjects: PostgreSQL data source objects to avoid
      backfilling.
    sqlServerExcludedObjects: SQLServer data source objects to avoid
      backfilling
  """
    mysqlExcludedObjects = _messages.MessageField('MysqlRdbms', 1)
    oracleExcludedObjects = _messages.MessageField('OracleRdbms', 2)
    postgresqlExcludedObjects = _messages.MessageField('PostgresqlRdbms', 3)
    sqlServerExcludedObjects = _messages.MessageField('SqlServerRdbms', 4)