from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2DetectIntentRequest(_messages.Message):
    """The request to detect user's intent.

  Fields:
    inputAudio: The natural language speech audio to be processed. This field
      should be populated iff `query_input` is set to an input audio config. A
      single request can contain up to 1 minute of speech audio data.
    outputAudioConfig: Instructs the speech synthesizer how to generate the
      output audio. If this field is not set and agent-level speech
      synthesizer is not configured, no output audio is generated.
    outputAudioConfigMask: Mask for output_audio_config indicating which
      settings in this request-level config should override speech synthesizer
      settings defined at agent-level. If unspecified or empty,
      output_audio_config replaces the agent-level config in its entirety.
    queryInput: Required. The input specification. It can be set to: 1. an
      audio config which instructs the speech recognizer how to process the
      speech audio, 2. a conversational query in the form of text, or 3. an
      event that specifies which intent to trigger.
    queryParams: The parameters of this query.
  """
    inputAudio = _messages.BytesField(1)
    outputAudioConfig = _messages.MessageField('GoogleCloudDialogflowV2OutputAudioConfig', 2)
    outputAudioConfigMask = _messages.StringField(3)
    queryInput = _messages.MessageField('GoogleCloudDialogflowV2QueryInput', 4)
    queryParams = _messages.MessageField('GoogleCloudDialogflowV2QueryParameters', 5)