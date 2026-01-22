from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LanguageCodePair(_messages.Message):
    """Used with unidirectional glossaries.

  Fields:
    sourceLanguageCode: Required. The BCP-47 language code of the input text,
      for example, "en-US". Expected to be an exact match for
      GlossaryTerm.language_code.
    targetLanguageCode: Required. The BCP-47 language code for translation
      output, for example, "zh-CN". Expected to be an exact match for
      GlossaryTerm.language_code.
  """
    sourceLanguageCode = _messages.StringField(1)
    targetLanguageCode = _messages.StringField(2)