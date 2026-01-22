from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2TextToSpeechSettings(_messages.Message):
    """Instructs the speech synthesizer on how to generate the output audio
  content.

  Enums:
    OutputAudioEncodingValueValuesEnum: Required. Audio encoding of the
      synthesized audio content.

  Messages:
    SynthesizeSpeechConfigsValue: Optional. Configuration of how speech should
      be synthesized, mapping from language
      (https://cloud.google.com/dialogflow/docs/reference/language) to
      SynthesizeSpeechConfig.

  Fields:
    enableTextToSpeech: Optional. Indicates whether text to speech is enabled.
      Even when this field is false, other settings in this proto are still
      retained.
    outputAudioEncoding: Required. Audio encoding of the synthesized audio
      content.
    sampleRateHertz: Optional. The synthesis sample rate (in hertz) for this
      audio. If not provided, then the synthesizer will use the default sample
      rate based on the audio encoding. If this is different from the voice's
      natural sample rate, then the synthesizer will honor this request by
      converting to the desired sample rate (which might result in worse audio
      quality).
    synthesizeSpeechConfigs: Optional. Configuration of how speech should be
      synthesized, mapping from language
      (https://cloud.google.com/dialogflow/docs/reference/language) to
      SynthesizeSpeechConfig.
  """

    class OutputAudioEncodingValueValuesEnum(_messages.Enum):
        """Required. Audio encoding of the synthesized audio content.

    Values:
      OUTPUT_AUDIO_ENCODING_UNSPECIFIED: Not specified.
      OUTPUT_AUDIO_ENCODING_LINEAR_16: Uncompressed 16-bit signed little-
        endian samples (Linear PCM). Audio content returned as LINEAR16 also
        contains a WAV header.
      OUTPUT_AUDIO_ENCODING_MP3: MP3 audio at 32kbps.
      OUTPUT_AUDIO_ENCODING_MP3_64_KBPS: MP3 audio at 64kbps.
      OUTPUT_AUDIO_ENCODING_OGG_OPUS: Opus encoded audio wrapped in an ogg
        container. The result will be a file which can be played natively on
        Android, and in browsers (at least Chrome and Firefox). The quality of
        the encoding is considerably higher than MP3 while using approximately
        the same bitrate.
      OUTPUT_AUDIO_ENCODING_MULAW: 8-bit samples that compand 14-bit audio
        samples using G.711 PCMU/mu-law.
    """
        OUTPUT_AUDIO_ENCODING_UNSPECIFIED = 0
        OUTPUT_AUDIO_ENCODING_LINEAR_16 = 1
        OUTPUT_AUDIO_ENCODING_MP3 = 2
        OUTPUT_AUDIO_ENCODING_MP3_64_KBPS = 3
        OUTPUT_AUDIO_ENCODING_OGG_OPUS = 4
        OUTPUT_AUDIO_ENCODING_MULAW = 5

    @encoding.MapUnrecognizedFields('additionalProperties')
    class SynthesizeSpeechConfigsValue(_messages.Message):
        """Optional. Configuration of how speech should be synthesized, mapping
    from language
    (https://cloud.google.com/dialogflow/docs/reference/language) to
    SynthesizeSpeechConfig.

    Messages:
      AdditionalProperty: An additional property for a
        SynthesizeSpeechConfigsValue object.

    Fields:
      additionalProperties: Additional properties of type
        SynthesizeSpeechConfigsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a SynthesizeSpeechConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudDialogflowV2SynthesizeSpeechConfig attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudDialogflowV2SynthesizeSpeechConfig', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    enableTextToSpeech = _messages.BooleanField(1)
    outputAudioEncoding = _messages.EnumField('OutputAudioEncodingValueValuesEnum', 2)
    sampleRateHertz = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    synthesizeSpeechConfigs = _messages.MessageField('SynthesizeSpeechConfigsValue', 4)