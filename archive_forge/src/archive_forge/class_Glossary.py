from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Glossary(_messages.Message):
    """Represents a glossary built from user provided data.

  Fields:
    endTime: Output only. When the glossary creation was finished.
    entryCount: Output only. The number of entries defined in the glossary.
    inputConfig: Required. Provides examples to build the glossary from. Total
      glossary must not exceed 10M Unicode codepoints.
    languageCodesSet: Used with equivalent term set glossaries.
    languagePair: Used with unidirectional glossaries.
    name: Required. The resource name of the glossary. Glossary names have the
      form `projects/{project-number-or-id}/locations/{location-
      id}/glossaries/{glossary-id}`.
    submitTime: Output only. When CreateGlossary was called.
  """
    endTime = _messages.StringField(1)
    entryCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    inputConfig = _messages.MessageField('GlossaryInputConfig', 3)
    languageCodesSet = _messages.MessageField('LanguageCodesSet', 4)
    languagePair = _messages.MessageField('LanguageCodePair', 5)
    name = _messages.StringField(6)
    submitTime = _messages.StringField(7)