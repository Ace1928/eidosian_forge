from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackupdrProjectsLocationsBackupVaultsDataSourcesBackupsGetRequest(_messages.Message):
    """A BackupdrProjectsLocationsBackupVaultsDataSourcesBackupsGetRequest
  object.

  Fields:
    name: Required. Name of the data source resource name, in the format `proj
      ects/{project_id}/locations/{location}/backupVaults/{backupVault}/dataSo
      urces/{datasource}/backups/{backup}`
  """
    name = _messages.StringField(1, required=True)