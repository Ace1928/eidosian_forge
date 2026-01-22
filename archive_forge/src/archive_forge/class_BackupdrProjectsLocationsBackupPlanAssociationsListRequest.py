from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackupdrProjectsLocationsBackupPlanAssociationsListRequest(_messages.Message):
    """A BackupdrProjectsLocationsBackupPlanAssociationsListRequest object.

  Fields:
    filter: Optional. Filtering results
    pageSize: Optional. Requested page size. Server may return fewer items
      than requested. If unspecified, server will pick an appropriate default.
    pageToken: Optional. A token identifying a page of results the server
      should return.
    parent: Required. The project and location for which to retrieve backup
      Plan Associations information, in the format
      `projects/{project_id}/locations/{location}`. In Cloud BackupDR,
      locations map to GCP regions, for example **us-central1**. To retrieve
      backup plan associations for all locations, use "-" for the `{location}`
      value.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)