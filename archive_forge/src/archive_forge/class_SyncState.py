from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SyncState(_messages.Message):
    """State indicating an ACM's progress syncing configurations to a cluster

  Enums:
    CodeValueValuesEnum: Sync status code

  Fields:
    code: Sync status code
    errors: A list of errors resulting from problematic configs. This list
      will be truncated after 100 errors, although it is unlikely for that
      many errors to simultaneously exist.
    importToken: Token indicating the state of the importer.
    lastSync: Timestamp of when ACM last successfully synced the repo The time
      format is specified in https://golang.org/pkg/time/#Time.String This
      field is being deprecated. Use last_sync_time instead.
    lastSyncTime: Timestamp type of when ACM last successfully synced the repo
    sourceToken: Token indicating the state of the repo.
    syncToken: Token indicating the state of the syncer.
  """

    class CodeValueValuesEnum(_messages.Enum):
        """Sync status code

    Values:
      SYNC_CODE_UNSPECIFIED: Config Sync cannot determine a sync code
      SYNCED: Config Sync successfully synced the git Repo with the cluster
      PENDING: Config Sync is in the progress of syncing a new change
      ERROR: Indicates an error configuring Config Sync, and user action is
        required
      NOT_CONFIGURED: Config Sync has been installed but not configured
      NOT_INSTALLED: Config Sync has not been installed
      UNAUTHORIZED: Error authorizing with the cluster
      UNREACHABLE: Cluster could not be reached
    """
        SYNC_CODE_UNSPECIFIED = 0
        SYNCED = 1
        PENDING = 2
        ERROR = 3
        NOT_CONFIGURED = 4
        NOT_INSTALLED = 5
        UNAUTHORIZED = 6
        UNREACHABLE = 7
    code = _messages.EnumField('CodeValueValuesEnum', 1)
    errors = _messages.MessageField('SyncError', 2, repeated=True)
    importToken = _messages.StringField(3)
    lastSync = _messages.StringField(4)
    lastSyncTime = _messages.StringField(5)
    sourceToken = _messages.StringField(6)
    syncToken = _messages.StringField(7)