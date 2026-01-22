from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RestoreServiceRequest(_messages.Message):
    """Request message for DataprocMetastore.Restore.

  Enums:
    RestoreTypeValueValuesEnum: Optional. The type of restore. If unspecified,
      defaults to METADATA_ONLY.

  Fields:
    backup: Optional. The relative resource name of the metastore service
      backup to restore from, in the following form:projects/{project_id}/loca
      tions/{location_id}/services/{service_id}/backups/{backup_id}. Mutually
      exclusive with backup_location, and exactly one of the two must be set.
    backupLocation: Optional. A Cloud Storage URI specifying the location of
      the backup artifacts, namely - backup avro files under "avro/",
      backup_metastore.json and service.json, in the following form:gs://.
      Mutually exclusive with backup, and exactly one of the two must be set.
    requestId: Optional. A request ID. Specify a unique request ID to allow
      the server to ignore the request if it has completed. The server will
      ignore subsequent requests that provide a duplicate request ID for at
      least 60 minutes after the first request.For example, if an initial
      request times out, followed by another request with the same request ID,
      the server ignores the second request to prevent the creation of
      duplicate commitments.The request ID must be a valid UUID
      (https://en.wikipedia.org/wiki/Universally_unique_identifier#Format). A
      zero UUID (00000000-0000-0000-0000-000000000000) is not supported.
    restoreType: Optional. The type of restore. If unspecified, defaults to
      METADATA_ONLY.
  """

    class RestoreTypeValueValuesEnum(_messages.Enum):
        """Optional. The type of restore. If unspecified, defaults to
    METADATA_ONLY.

    Values:
      RESTORE_TYPE_UNSPECIFIED: The restore type is unknown.
      FULL: The service's metadata and configuration are restored.
      METADATA_ONLY: Only the service's metadata is restored.
    """
        RESTORE_TYPE_UNSPECIFIED = 0
        FULL = 1
        METADATA_ONLY = 2
    backup = _messages.StringField(1)
    backupLocation = _messages.StringField(2)
    requestId = _messages.StringField(3)
    restoreType = _messages.EnumField('RestoreTypeValueValuesEnum', 4)