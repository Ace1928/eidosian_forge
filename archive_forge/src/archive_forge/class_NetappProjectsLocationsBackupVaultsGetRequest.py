from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsBackupVaultsGetRequest(_messages.Message):
    """A NetappProjectsLocationsBackupVaultsGetRequest object.

  Fields:
    name: Required. The backupVault resource name, in the format `projects/{pr
      oject_id}/locations/{location}/backupVaults/{backup_vault_id}`
  """
    name = _messages.StringField(1, required=True)