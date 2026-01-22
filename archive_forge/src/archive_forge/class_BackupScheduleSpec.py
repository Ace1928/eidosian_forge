from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackupScheduleSpec(_messages.Message):
    """Defines specifications of the backup schedule.

  Fields:
    cronSpec: Cron style schedule specification.
  """
    cronSpec = _messages.MessageField('CrontabSpec', 1)