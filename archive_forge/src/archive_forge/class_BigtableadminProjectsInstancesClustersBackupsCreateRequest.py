from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigtableadminProjectsInstancesClustersBackupsCreateRequest(_messages.Message):
    """A BigtableadminProjectsInstancesClustersBackupsCreateRequest object.

  Fields:
    backup: A Backup resource to be passed as the request body.
    backupId: Required. The id of the backup to be created. The `backup_id`
      along with the parent `parent` are combined as
      {parent}/backups/{backup_id} to create the full backup name, of the
      form: `projects/{project}/instances/{instance}/clusters/{cluster}/backup
      s/{backup_id}`. This string must be between 1 and 50 characters in
      length and match the regex _a-zA-Z0-9*.
    parent: Required. This must be one of the clusters in the instance in
      which this table is located. The backup will be stored in this cluster.
      Values are of the form
      `projects/{project}/instances/{instance}/clusters/{cluster}`.
  """
    backup = _messages.MessageField('Backup', 1)
    backupId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)