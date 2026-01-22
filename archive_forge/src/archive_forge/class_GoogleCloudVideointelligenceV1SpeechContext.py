from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1SpeechContext(_messages.Message):
    """Provides "hints" to the speech recognizer to favor specific words and
  phrases in the results.

  Fields:
    phrases: Optional. A list of strings containing words and phrases "hints"
      so that the speech recognition is more likely to recognize them. This
      can be used to improve the accuracy for specific words and phrases, for
      example, if specific commands are typically spoken by the user. This can
      also be used to add additional words to the vocabulary of the
      recognizer. See [usage
      limits](https://cloud.google.com/speech/limits#content).
  """
    phrases = _messages.StringField(1, repeated=True)