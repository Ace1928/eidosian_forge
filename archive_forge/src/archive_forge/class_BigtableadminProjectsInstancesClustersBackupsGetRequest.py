from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesClustersBackupsGetRequest(_messages.Message):
    """A BigtableadminProjectsInstancesClustersBackupsGetRequest object.

  Fields:
    name: Required. Name of the backup. Values are of the form `projects/{proj
      ect}/instances/{instance}/clusters/{cluster}/backups/{backup}`.
  """
    name = _messages.StringField(1, required=True)