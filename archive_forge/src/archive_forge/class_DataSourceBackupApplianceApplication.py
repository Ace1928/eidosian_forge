from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataSourceBackupApplianceApplication(_messages.Message):
    """BackupApplianceApplication describes a Source Resource when it is an
  application backed up by a BackupAppliance.

  Fields:
    applianceId: Appliance Id of the Backup Appliance.
    applicationId: The appid field of the application within the Backup
      Appliance.
    applicationName: The name of the Application as known to the Backup
      Appliance.
    backupAppliance: Appliance name.
    hostId: Hostid of the application host.
    hostname: Hostname of the host where the application is running.
    type: The type of the application. e.g. VMBackup
  """
    applianceId = _messages.IntegerField(1)
    applicationId = _messages.IntegerField(2)
    applicationName = _messages.StringField(3)
    backupAppliance = _messages.StringField(4)
    hostId = _messages.IntegerField(5)
    hostname = _messages.StringField(6)
    type = _messages.StringField(7)