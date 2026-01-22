from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateTextGlossaryConfig(_messages.Message):
    """Configures which glossary should be used for a specific target language,
  and defines options for applying that glossary.

  Fields:
    glossary: Required. Specifies the glossary used for this translation. Use
      this format: projects/*/locations/*/glossaries/*
    ignoreCase: Optional. Indicates match is case-insensitive. Default value
      is false if missing.
  """
    glossary = _messages.StringField(1)
    ignoreCase = _messages.BooleanField(2)