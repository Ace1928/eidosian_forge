from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1BigQueryDateShardedSpec(_messages.Message):
    """Specification for a group of BigQuery tables with the `[prefix]YYYYMMDD`
  name pattern. For more information, see [Introduction to partitioned tables]
  (https://cloud.google.com/bigquery/docs/partitioned-
  tables#partitioning_versus_sharding).

  Fields:
    dataset: Output only. The Data Catalog resource name of the dataset entry
      the current table belongs to. For example: `projects/{PROJECT_ID}/locati
      ons/{LOCATION}/entrygroups/{ENTRY_GROUP_ID}/entries/{ENTRY_ID}`.
    latestShardResource: Output only. BigQuery resource name of the latest
      shard.
    shardCount: Output only. Total number of shards.
    tablePrefix: Output only. The table name prefix of the shards. The name of
      any given shard is `[table_prefix]YYYYMMDD`. For example, for the
      `MyTable20180101` shard, the `table_prefix` is `MyTable`.
  """
    dataset = _messages.StringField(1)
    latestShardResource = _messages.StringField(2)
    shardCount = _messages.IntegerField(3)
    tablePrefix = _messages.StringField(4)