from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MysqlProfile(_messages.Message):
    """MySQL database profile.

  Fields:
    hostname: Required. Hostname for the MySQL connection.
    password: Required. Input only. Password for the MySQL connection.
    port: Port for the MySQL connection, default value is 3306.
    sslConfig: SSL configuration for the MySQL connection.
    username: Required. Username for the MySQL connection.
  """
    hostname = _messages.StringField(1)
    password = _messages.StringField(2)
    port = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    sslConfig = _messages.MessageField('MysqlSslConfig', 4)
    username = _messages.StringField(5)