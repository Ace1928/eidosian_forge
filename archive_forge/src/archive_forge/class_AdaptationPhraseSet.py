from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AdaptationPhraseSet(_messages.Message):
    """A biasing PhraseSet, which can be either a string referencing the name
  of an existing PhraseSets resource, or an inline definition of a PhraseSet.

  Fields:
    inlinePhraseSet: An inline defined PhraseSet.
    phraseSet: The name of an existing PhraseSet resource. The user must have
      read access to the resource and it must not be deleted.
  """
    inlinePhraseSet = _messages.MessageField('PhraseSet', 1)
    phraseSet = _messages.StringField(2)