from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MysqlRdbms(_messages.Message):
    """MySQL database structure

  Fields:
    mysqlDatabases: Mysql databases on the server
  """
    mysqlDatabases = _messages.MessageField('MysqlDatabase', 1, repeated=True)