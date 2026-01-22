from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2ReplaceDictionaryConfig(_messages.Message):
    """Replace each input value with a value randomly selected from the
  dictionary.

  Fields:
    wordList: A list of words to select from for random replacement. The
      [limits](https://cloud.google.com/sensitive-data-protection/limits) page
      contains details about the size limits of dictionaries.
  """
    wordList = _messages.MessageField('GooglePrivacyDlpV2WordList', 1)