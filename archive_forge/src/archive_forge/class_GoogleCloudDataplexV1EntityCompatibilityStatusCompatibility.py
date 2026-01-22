from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1EntityCompatibilityStatusCompatibility(_messages.Message):
    """Provides compatibility information for a specific metadata store.

  Fields:
    compatible: Output only. Whether the entity is compatible and can be
      represented in the metadata store.
    reason: Output only. Provides additional detail if the entity is
      incompatible with the metadata store.
  """
    compatible = _messages.BooleanField(1)
    reason = _messages.StringField(2)