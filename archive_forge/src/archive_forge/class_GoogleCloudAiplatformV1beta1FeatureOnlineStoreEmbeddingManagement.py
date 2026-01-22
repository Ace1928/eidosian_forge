from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1FeatureOnlineStoreEmbeddingManagement(_messages.Message):
    """Deprecated: This sub message is no longer needed anymore and embedding
  management is automatically enabled when specifying Optimized storage type.
  Contains settings for embedding management.

  Fields:
    enabled: Optional. Immutable. Whether to enable embedding management in
      this FeatureOnlineStore. It's immutable after creation to ensure the
      FeatureOnlineStore availability.
  """
    enabled = _messages.BooleanField(1)