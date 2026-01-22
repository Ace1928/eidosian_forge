from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TextProperty(_messages.Message):
    """Additional information detected on the structural component.

  Fields:
    detectedBreak: Detected start or end of a text segment.
    detectedLanguages: A list of detected languages together with confidence.
  """
    detectedBreak = _messages.MessageField('DetectedBreak', 1)
    detectedLanguages = _messages.MessageField('DetectedLanguage', 2, repeated=True)