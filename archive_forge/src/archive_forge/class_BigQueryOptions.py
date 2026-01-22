from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigQueryOptions(_messages.Message):
    """Options that change functionality of a sink exporting data to BigQuery.

  Fields:
    usePartitionedTables: Optional. Whether to use BigQuery's partition tables
      (https://cloud.google.com/bigquery/docs/partitioned-tables). By default,
      Cloud Logging creates dated tables based on the log entries' timestamps,
      e.g. syslog_20170523. With partitioned tables the date suffix is no
      longer present and special query syntax
      (https://cloud.google.com/bigquery/docs/querying-partitioned-tables) has
      to be used instead. In both cases, tables are sharded based on UTC
      timezone.
    usesTimestampColumnPartitioning: Output only. True if new timestamp column
      based partitioning is in use, false if legacy ingress-time partitioning
      is in use.All new sinks will have this field set true and will use
      timestamp column based partitioning. If use_partitioned_tables is false,
      this value has no meaning and will be false. Legacy sinks using
      partitioned tables will have this field set to false.
  """
    usePartitionedTables = _messages.BooleanField(1)
    usesTimestampColumnPartitioning = _messages.BooleanField(2)