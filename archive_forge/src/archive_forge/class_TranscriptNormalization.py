from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranscriptNormalization(_messages.Message):
    """Transcription normalization configuration. Use transcription
  normalization to automatically replace parts of the transcript with phrases
  of your choosing. For StreamingRecognize, this normalization only applies to
  stable partial transcripts (stability > 0.8) and final transcripts.

  Fields:
    entries: A list of replacement entries. We will perform replacement with
      one entry at a time. For example, the second entry in ["cat" => "dog",
      "mountain cat" => "mountain dog"] will never be applied because we will
      always process the first entry before it. At most 100 entries.
  """
    entries = _messages.MessageField('Entry', 1, repeated=True)