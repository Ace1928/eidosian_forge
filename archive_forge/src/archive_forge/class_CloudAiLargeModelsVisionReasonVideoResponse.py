from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiLargeModelsVisionReasonVideoResponse(_messages.Message):
    """Video reasoning response.

  Fields:
    responses: Generated text responses. The generated responses for different
      segments within the same video.
  """
    responses = _messages.MessageField('CloudAiLargeModelsVisionReasonVideoResponseTextResponse', 1, repeated=True)