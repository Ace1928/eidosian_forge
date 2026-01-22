from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackupApplianceBackupProperties(_messages.Message):
    """BackupApplianceBackupProperties represents Compute Engine instance
  backup properties.

  Fields:
    finalizeTime: Output only. The time when this backup object was finalized
      (if none, backup is not finalized).
    generationId: Output only. The numeric generation ID of the backup
      (monotonically increasing).
    recoveryRangeEndTime: Optional. The latest timestamp of data available in
      this Backup.
    recoveryRangeStartTime: Optional. The earliest timestamp of data available
      in this Backup.
  """
    finalizeTime = _messages.StringField(1)
    generationId = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    recoveryRangeEndTime = _messages.StringField(3)
    recoveryRangeStartTime = _messages.StringField(4)