from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LanguageCodesSet(_messages.Message):
    """Used with equivalent term set glossaries.

  Fields:
    languageCodes: The BCP-47 language code(s) for terms defined in the
      glossary. All entries are unique. The list contains at least two
      entries. Expected to be an exact match for GlossaryTerm.language_code.
  """
    languageCodes = _messages.StringField(1, repeated=True)