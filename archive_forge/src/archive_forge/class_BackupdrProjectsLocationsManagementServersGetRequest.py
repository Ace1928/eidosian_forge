from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackupdrProjectsLocationsManagementServersGetRequest(_messages.Message):
    """A BackupdrProjectsLocationsManagementServersGetRequest object.

  Fields:
    name: Required. Name of the management server resource name, in the format
      `projects/{project_id}/locations/{location}/managementServers/{resource_
      name}`
  """
    name = _messages.StringField(1, required=True)