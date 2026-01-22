from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetastoreProjectsLocationsServicesBackupsGetRequest(_messages.Message):
    """A MetastoreProjectsLocationsServicesBackupsGetRequest object.

  Fields:
    name: Required. The relative resource name of the backup to retrieve, in
      the following form:projects/{project_number}/locations/{location_id}/ser
      vices/{service_id}/backups/{backup_id}.
  """
    name = _messages.StringField(1, required=True)