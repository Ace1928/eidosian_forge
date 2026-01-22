from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiLargeModelsVisionEmbedVideoResponse(_messages.Message):
    """Video embedding response.

  Fields:
    videoEmbeddings: The embedding vector for the video.
  """
    videoEmbeddings = _messages.MessageField('extra_types.JsonValue', 1, repeated=True)