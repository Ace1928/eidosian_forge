from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MySqlConnectionProfile(_messages.Message):
    """Specifies connection parameters required specifically for MySQL
  databases.

  Fields:
    cloudSqlId: If the source is a Cloud SQL database, use this field to
      provide the Cloud SQL instance ID of the source.
    hasPassword: Output only. Indicates If this connection profile password is
      stored.
    host: Required. The IP or hostname of the source MySQL database.
    password: Required. Input only. The password for the user that Database
      Migration Service will be using to connect to the database. This field
      is not returned on request, and the value is encrypted when stored in
      Database Migration Service.
    port: Required. The network port of the source MySQL database.
    ssl: SSL configuration for the destination to connect to the source
      database.
    username: Required. The username that Database Migration Service will use
      to connect to the database. The value is encrypted when stored in
      Database Migration Service.
  """
    cloudSqlId = _messages.StringField(1)
    hasPassword = _messages.BooleanField(2)
    host = _messages.StringField(3)
    password = _messages.StringField(4)
    port = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    ssl = _messages.MessageField('SslConfig', 6)
    username = _messages.StringField(7)