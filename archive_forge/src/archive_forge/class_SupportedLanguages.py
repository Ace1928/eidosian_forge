from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SupportedLanguages(_messages.Message):
    """The response message for discovering supported languages.

  Fields:
    languages: A list of supported language responses. This list contains an
      entry for each language the Translation API supports.
  """
    languages = _messages.MessageField('SupportedLanguage', 1, repeated=True)