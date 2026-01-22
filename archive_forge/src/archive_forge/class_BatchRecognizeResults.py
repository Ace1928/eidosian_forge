from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BatchRecognizeResults(_messages.Message):
    """Output type for Cloud Storage of BatchRecognize transcripts. Though this
  proto isn't returned in this API anywhere, the Cloud Storage transcripts
  will be this proto serialized and should be parsed as such.

  Fields:
    metadata: Metadata about the recognition.
    results: Sequential list of transcription results corresponding to
      sequential portions of audio.
  """
    metadata = _messages.MessageField('RecognitionResponseMetadata', 1)
    results = _messages.MessageField('SpeechRecognitionResult', 2, repeated=True)