from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedidentitiesProjectsLocationsGlobalDomainsBackupsPatchRequest(_messages.Message):
    """A ManagedidentitiesProjectsLocationsGlobalDomainsBackupsPatchRequest
  object.

  Fields:
    backup: A Backup resource to be passed as the request body.
    name: Output only. The unique name of the Backup in the form of `projects/
      {project_id}/locations/global/domains/{domain_name}/backups/{name}`
    updateMask: Required. Mask of fields to update. At least one path must be
      supplied in this field. The elements of the repeated paths field may
      only include these fields from Backup: * `labels`
  """
    backup = _messages.MessageField('Backup', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)