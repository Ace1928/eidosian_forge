from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiLargeModelsVisionVideo(_messages.Message):
    """Video

  Fields:
    uri: Path to another storage (typically Google Cloud Storage).
    video: Raw bytes.
  """
    uri = _messages.StringField(1)
    video = _messages.BytesField(2)