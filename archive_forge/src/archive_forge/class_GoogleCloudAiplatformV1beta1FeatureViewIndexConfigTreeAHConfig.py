from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1FeatureViewIndexConfigTreeAHConfig(_messages.Message):
    """Configuration options for the tree-AH algorithm.

  Fields:
    leafNodeEmbeddingCount: Optional. Number of embeddings on each leaf node.
      The default value is 1000 if not set.
  """
    leafNodeEmbeddingCount = _messages.IntegerField(1)