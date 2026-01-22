from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileProjectsLocationsBackupsGetRequest(_messages.Message):
    """A FileProjectsLocationsBackupsGetRequest object.

  Fields:
    name: Required. The backup resource name, in the format
      `projects/{project_number}/locations/{location}/backups/{backup_id}`.
  """
    name = _messages.StringField(1, required=True)