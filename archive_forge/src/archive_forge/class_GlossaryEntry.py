from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GlossaryEntry(_messages.Message):
    """Represents a single entry in a glossary.

  Fields:
    description: Describes the glossary entry.
    name: Required. The resource name of the entry. Format:
      "projects/*/locations/*/glossaries/*/glossaryEntries/*"
    termsPair: Used for an unidirectional glossary.
    termsSet: Used for an equivalent term sets glossary.
  """
    description = _messages.StringField(1)
    name = _messages.StringField(2)
    termsPair = _messages.MessageField('GlossaryTermsPair', 3)
    termsSet = _messages.MessageField('GlossaryTermsSet', 4)