from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackupdrProjectsLocationsBackupPlanAssociationsGetRequest(_messages.Message):
    """A BackupdrProjectsLocationsBackupPlanAssociationsGetRequest object.

  Fields:
    name: Required. Name of the backup plan association resource, in the
      format `projects/{project}/locations/{location}/backupPlanAssociations/{
      backupPlanAssociationId}`
  """
    name = _messages.StringField(1, required=True)