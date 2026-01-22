from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackupInfo(_messages.Message):
    """Information about a backup.

  Fields:
    backup: Name of the backup.
    createTime: The time the CreateBackup request was received.
    sourceDatabase: Name of the database the backup was created from.
    versionTime: The backup contains an externally consistent copy of
      `source_database` at the timestamp specified by `version_time`. If the
      CreateBackup request did not specify `version_time`, the `version_time`
      of the backup is equivalent to the `create_time`.
  """
    backup = _messages.StringField(1)
    createTime = _messages.StringField(2)
    sourceDatabase = _messages.StringField(3)
    versionTime = _messages.StringField(4)