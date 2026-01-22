from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3AdvancedSettingsSpeechSettings(_messages.Message):
    """Define behaviors of speech to text detection.

  Messages:
    ModelsValue: Mapping from language to Speech-to-Text model. The mapped
      Speech-to-Text model will be selected for requests from its
      corresponding language. For more information, see [Speech
      models](https://cloud.google.com/dialogflow/cx/docs/concept/speech-
      models).

  Fields:
    endpointerSensitivity: Sensitivity of the speech model that detects the
      end of speech. Scale from 0 to 100.
    models: Mapping from language to Speech-to-Text model. The mapped Speech-
      to-Text model will be selected for requests from its corresponding
      language. For more information, see [Speech
      models](https://cloud.google.com/dialogflow/cx/docs/concept/speech-
      models).
    noSpeechTimeout: Timeout before detecting no speech.
    useTimeoutBasedEndpointing: Use timeout based endpointing, interpreting
      endpointer sensitivy as seconds of timeout value.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ModelsValue(_messages.Message):
        """Mapping from language to Speech-to-Text model. The mapped Speech-to-
    Text model will be selected for requests from its corresponding language.
    For more information, see [Speech
    models](https://cloud.google.com/dialogflow/cx/docs/concept/speech-
    models).

    Messages:
      AdditionalProperty: An additional property for a ModelsValue object.

    Fields:
      additionalProperties: Additional properties of type ModelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ModelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    endpointerSensitivity = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    models = _messages.MessageField('ModelsValue', 2)
    noSpeechTimeout = _messages.StringField(3)
    useTimeoutBasedEndpointing = _messages.BooleanField(4)