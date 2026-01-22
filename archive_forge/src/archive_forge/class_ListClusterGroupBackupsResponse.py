from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListClusterGroupBackupsResponse(_messages.Message):
    """A list of cluster group backups.

  Fields:
    clusterGroupBackups: A list of cluster group backups.
    nextPageToken: A token that you can send as `page_token` to retrieve the
      next page. If you omit this field, there are no subsequent pages.
    unreachable: List of locations that could not be reached.
  """
    clusterGroupBackups = _messages.MessageField('ClusterGroupBackup', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)