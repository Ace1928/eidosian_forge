from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Database(_messages.Message):
    """A Cloud Spanner database.

  Enums:
    DatabaseDialectValueValuesEnum: Output only. The dialect of the Cloud
      Spanner Database.
    StateValueValuesEnum: Output only. The current database state.

  Fields:
    createTime: Output only. If exists, the time at which the database
      creation started.
    databaseDialect: Output only. The dialect of the Cloud Spanner Database.
    defaultLeader: Output only. The read-write region which contains the
      database's leader replicas. This is the same as the value of
      default_leader database option set using DatabaseAdmin.CreateDatabase or
      DatabaseAdmin.UpdateDatabaseDdl. If not explicitly set, this is empty.
    earliestVersionTime: Output only. Earliest timestamp at which older
      versions of the data can be read. This value is continuously updated by
      Cloud Spanner and becomes stale the moment it is queried. If you are
      using this value to recover data, make sure to account for the time from
      the moment when the value is queried to the moment when you initiate the
      recovery.
    enableDropProtection: Whether drop protection is enabled for this
      database. Defaults to false, if not set. For more details, please see
      how to [prevent accidental database
      deletion](https://cloud.google.com/spanner/docs/prevent-database-
      deletion).
    encryptionConfig: Output only. For databases that are using customer
      managed encryption, this field contains the encryption configuration for
      the database. For databases that are using Google default or other types
      of encryption, this field is empty.
    encryptionInfo: Output only. For databases that are using customer managed
      encryption, this field contains the encryption information for the
      database, such as all Cloud KMS key versions that are in use. The
      `encryption_status' field inside of each `EncryptionInfo` is not
      populated. For databases that are using Google default or other types of
      encryption, this field is empty. This field is propagated lazily from
      the backend. There might be a delay from when a key version is being
      used and when it appears in this field.
    name: Required. The name of the database. Values are of the form
      `projects//instances//databases/`, where `` is as specified in the
      `CREATE DATABASE` statement. This name can be passed to other API
      methods to identify the database.
    quorumInfo: Output only. Applicable only for databases that use dual
      region instance configurations. Contains information about the quorum.
    reconciling: Output only. If true, the database is being updated. If
      false, there are no ongoing update operations for the database.
    restoreInfo: Output only. Applicable only for restored databases. Contains
      information about the restore source.
    state: Output only. The current database state.
    versionRetentionPeriod: Output only. The period in which Cloud Spanner
      retains all versions of data for the database. This is the same as the
      value of version_retention_period database option set using
      UpdateDatabaseDdl. Defaults to 1 hour, if not set.
  """

    class DatabaseDialectValueValuesEnum(_messages.Enum):
        """Output only. The dialect of the Cloud Spanner Database.

    Values:
      DATABASE_DIALECT_UNSPECIFIED: Default value. This value will create a
        database with the GOOGLE_STANDARD_SQL dialect.
      GOOGLE_STANDARD_SQL: GoogleSQL supported SQL.
      POSTGRESQL: PostgreSQL supported SQL.
    """
        DATABASE_DIALECT_UNSPECIFIED = 0
        GOOGLE_STANDARD_SQL = 1
        POSTGRESQL = 2

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current database state.

    Values:
      STATE_UNSPECIFIED: Not specified.
      CREATING: The database is still being created. Operations on the
        database may fail with `FAILED_PRECONDITION` in this state.
      READY: The database is fully created and ready for use.
      READY_OPTIMIZING: The database is fully created and ready for use, but
        is still being optimized for performance and cannot handle full load.
        In this state, the database still references the backup it was restore
        from, preventing the backup from being deleted. When optimizations are
        complete, the full performance of the database will be restored, and
        the database will transition to `READY` state.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        READY = 2
        READY_OPTIMIZING = 3
    createTime = _messages.StringField(1)
    databaseDialect = _messages.EnumField('DatabaseDialectValueValuesEnum', 2)
    defaultLeader = _messages.StringField(3)
    earliestVersionTime = _messages.StringField(4)
    enableDropProtection = _messages.BooleanField(5)
    encryptionConfig = _messages.MessageField('EncryptionConfig', 6)
    encryptionInfo = _messages.MessageField('EncryptionInfo', 7, repeated=True)
    name = _messages.StringField(8)
    quorumInfo = _messages.MessageField('QuorumInfo', 9)
    reconciling = _messages.BooleanField(10)
    restoreInfo = _messages.MessageField('RestoreInfo', 11)
    state = _messages.EnumField('StateValueValuesEnum', 12)
    versionRetentionPeriod = _messages.StringField(13)