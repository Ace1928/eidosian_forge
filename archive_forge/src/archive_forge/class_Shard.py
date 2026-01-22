from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class Shard(_messages.Message):
    """Output only. Details about the shard.

  Fields:
    estimatedShardDuration: Output only. The estimated shard duration based on
      previous test case timing records, if available.
    numShards: Output only. The total number of shards.
    shardIndex: Output only. The index of the shard among all the shards.
    testTargetsForShard: Output only. Test targets for each shard. Only set
      for manual sharding.
  """
    estimatedShardDuration = _messages.StringField(1)
    numShards = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    shardIndex = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    testTargetsForShard = _messages.MessageField('TestTargetsForShard', 4)