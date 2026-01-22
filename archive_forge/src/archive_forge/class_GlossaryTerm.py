from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GlossaryTerm(_messages.Message):
    """Represents a single glossary term

  Fields:
    languageCode: The language for this glossary term.
    text: The text for the glossary term.
  """
    languageCode = _messages.StringField(1)
    text = _messages.StringField(2)