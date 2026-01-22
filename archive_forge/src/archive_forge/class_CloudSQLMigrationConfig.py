from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudSQLMigrationConfig(_messages.Message):
    """Configuration information for migrating from self-managed hive metastore
  on Google Cloud using Cloud SQL as the backend database to Dataproc
  Metastore.

  Fields:
    cdcConfig: Required. Configuration information to start the Change Data
      Capture (CDC) streams from customer database to backend database of
      Dataproc Metastore. Dataproc Metastore switches to using its backend
      database after the cutover phase of migration.
    cloudSqlConnectionConfig: Required. Configuration information to establish
      customer database connection before the cutover phase of migration
  """
    cdcConfig = _messages.MessageField('CdcConfig', 1)
    cloudSqlConnectionConfig = _messages.MessageField('CloudSQLConnectionConfig', 2)