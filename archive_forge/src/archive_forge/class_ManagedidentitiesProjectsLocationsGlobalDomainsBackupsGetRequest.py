from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedidentitiesProjectsLocationsGlobalDomainsBackupsGetRequest(_messages.Message):
    """A ManagedidentitiesProjectsLocationsGlobalDomainsBackupsGetRequest
  object.

  Fields:
    name: Required. The backup resource name using the form: `projects/{projec
      t_id}/locations/global/domains/{domain_name}/backups/{backup_id}`
  """
    name = _messages.StringField(1, required=True)