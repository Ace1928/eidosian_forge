from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageDatabasecenterPartnerapiV1mainDatabaseMetadata(_messages.Message):
    """Metadata for individual databases created in an instance. i.e. spanner
  instance can have multiple databases with unique configuration settings.

  Fields:
    backupConfiguration: Backup configuration for this database
    backupRun: Information about the last backup attempt for this database
    product: A StorageDatabasecenterProtoCommonProduct attribute.
    resourceId: A StorageDatabasecenterPartnerapiV1mainDatabaseResourceId
      attribute.
    resourceName: Required. Database name. Resource name to follow CAIS
      resource_name format as noted here go/condor-common-datamodel
  """
    backupConfiguration = _messages.MessageField('StorageDatabasecenterPartnerapiV1mainBackupConfiguration', 1)
    backupRun = _messages.MessageField('StorageDatabasecenterPartnerapiV1mainBackupRun', 2)
    product = _messages.MessageField('StorageDatabasecenterProtoCommonProduct', 3)
    resourceId = _messages.MessageField('StorageDatabasecenterPartnerapiV1mainDatabaseResourceId', 4)
    resourceName = _messages.StringField(5)