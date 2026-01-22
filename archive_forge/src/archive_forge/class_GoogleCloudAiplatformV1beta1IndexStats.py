from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1IndexStats(_messages.Message):
    """Stats of the Index.

  Fields:
    shardsCount: Output only. The number of shards in the Index.
    vectorsCount: Output only. The number of vectors in the Index.
  """
    shardsCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    vectorsCount = _messages.IntegerField(2)