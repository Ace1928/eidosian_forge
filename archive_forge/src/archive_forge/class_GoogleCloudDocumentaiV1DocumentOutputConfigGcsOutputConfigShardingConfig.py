from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1DocumentOutputConfigGcsOutputConfigShardingConfig(_messages.Message):
    """The sharding config for the output document.

  Fields:
    pagesOverlap: The number of overlapping pages between consecutive shards.
    pagesPerShard: The number of pages per shard.
  """
    pagesOverlap = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pagesPerShard = _messages.IntegerField(2, variant=_messages.Variant.INT32)