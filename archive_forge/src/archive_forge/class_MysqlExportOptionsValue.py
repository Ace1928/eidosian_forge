from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class MysqlExportOptionsValue(_messages.Message):
    """Options for exporting from MySQL.

      Fields:
        masterData: Option to include SQL statement required to set up
          replication. If set to `1`, the dump file includes a CHANGE MASTER
          TO statement with the binary log coordinates, and --set-gtid-purged
          is set to ON. If set to `2`, the CHANGE MASTER TO statement is
          written as a SQL comment and has no effect. If set to any value
          other than `1`, --set-gtid-purged is set to OFF.
      """
    masterData = _messages.IntegerField(1, variant=_messages.Variant.INT32)