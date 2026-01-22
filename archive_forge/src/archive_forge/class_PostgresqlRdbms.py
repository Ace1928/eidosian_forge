from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PostgresqlRdbms(_messages.Message):
    """PostgreSQL database structure.

  Fields:
    postgresqlSchemas: PostgreSQL schemas in the database server.
  """
    postgresqlSchemas = _messages.MessageField('PostgresqlSchema', 1, repeated=True)