from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1NearestNeighborQueryEmbedding(_messages.Message):
    """The embedding vector.

  Fields:
    value: Optional. Individual value in the embedding.
  """
    value = _messages.FloatField(1, repeated=True, variant=_messages.Variant.FLOAT)