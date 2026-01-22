from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackupApplianceBackupConfig(_messages.Message):
    """BackupApplianceBackupConfig captures the backup configuration for
  applications that are protected by Backup Appliances.

  Fields:
    applicationName: The name of the application.
    backupApplianceId: The ID of the backup appliance.
    backupApplianceName: The name of the backup appliance.
    hostName: The name of the host where the application is running.
    slaId: The ID of the SLA of this application.
    slpName: The name of the SLP associated with the application.
    sltName: The name of the SLT associated with the application.
  """
    applicationName = _messages.StringField(1)
    backupApplianceId = _messages.IntegerField(2)
    backupApplianceName = _messages.StringField(3)
    hostName = _messages.StringField(4)
    slaId = _messages.IntegerField(5)
    slpName = _messages.StringField(6)
    sltName = _messages.StringField(7)