from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PostgresqlProfile(_messages.Message):
    """PostgreSQL database profile.

  Fields:
    database: Required. Database for the PostgreSQL connection.
    hostname: Required. Hostname for the PostgreSQL connection.
    password: Required. Password for the PostgreSQL connection.
    port: Port for the PostgreSQL connection, default value is 5432.
    username: Required. Username for the PostgreSQL connection.
  """
    database = _messages.StringField(1)
    hostname = _messages.StringField(2)
    password = _messages.StringField(3)
    port = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    username = _messages.StringField(5)