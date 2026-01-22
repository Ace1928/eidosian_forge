from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LongRunningRecognizeResponse(_messages.Message):
    """The only message returned to the client by the `LongRunningRecognize`
  method. It contains the result as zero or more sequential
  `SpeechRecognitionResult` messages. It is included in the `result.response`
  field of the `Operation` returned by the `GetOperation` call of the
  `google::longrunning::Operations` service.

  Fields:
    outputConfig: Original output config if present in the request.
    outputError: If the transcript output fails this field contains the
      relevant error.
    requestId: The ID associated with the request. This is a unique ID
      specific only to the given request.
    results: Sequential list of transcription results corresponding to
      sequential portions of audio.
    speechAdaptationInfo: Provides information on speech adaptation behavior
      in response
    totalBilledTime: When available, billed audio seconds for the
      corresponding request.
  """
    outputConfig = _messages.MessageField('TranscriptOutputConfig', 1)
    outputError = _messages.MessageField('Status', 2)
    requestId = _messages.IntegerField(3)
    results = _messages.MessageField('SpeechRecognitionResult', 4, repeated=True)
    speechAdaptationInfo = _messages.MessageField('SpeechAdaptationInfo', 5)
    totalBilledTime = _messages.StringField(6)