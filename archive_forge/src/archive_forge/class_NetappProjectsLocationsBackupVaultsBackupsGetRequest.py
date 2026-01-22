from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsBackupVaultsBackupsGetRequest(_messages.Message):
    """A NetappProjectsLocationsBackupVaultsBackupsGetRequest object.

  Fields:
    name: Required. The backup resource name, in the format `projects/{project
      _id}/locations/{location}/backupVaults/{backup_vault_id}/backups/{backup
      _id}`
  """
    name = _messages.StringField(1, required=True)