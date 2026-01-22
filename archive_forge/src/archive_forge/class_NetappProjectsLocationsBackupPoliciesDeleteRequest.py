from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetappProjectsLocationsBackupPoliciesDeleteRequest(_messages.Message):
    """A NetappProjectsLocationsBackupPoliciesDeleteRequest object.

  Fields:
    name: Required. The backup policy resource name, in the format `projects/{
      project_id}/locations/{location}/backupPolicies/{backup_policy_id}`
  """
    name = _messages.StringField(1, required=True)