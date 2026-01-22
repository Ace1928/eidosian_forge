from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiLargeModelsVisionMedia(_messages.Message):
    """Media.

  Fields:
    image: Image.
    video: Video
  """
    image = _messages.MessageField('CloudAiLargeModelsVisionImage', 1)
    video = _messages.MessageField('CloudAiLargeModelsVisionVideo', 2)