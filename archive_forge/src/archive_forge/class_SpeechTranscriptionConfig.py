from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpeechTranscriptionConfig(_messages.Message):
    """Configure transcription options for speech: keyword.

  Fields:
    languageCode: Language code to use for configuring `speech:` search
      keyword. If unset, the default language will be English (en-US). This
      language code will be validated under [BCP-47](https://www.rfc-
      editor.org/rfc/bcp/bcp47.txt). Example: "en-US". See [Language
      Support](https://cloud.google.com/speech/docs/languages) for a list of
      the currently supported language codes.
  """
    languageCode = _messages.StringField(1)