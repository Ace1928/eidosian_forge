from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NlpSaftLanguageSpanSequence(_messages.Message):
    """A NlpSaftLanguageSpanSequence object.

  Fields:
    languageSpans: A sequence of LanguageSpan objects, each assigning a
      language to a subspan of the input.
    probability: The probability of this sequence of LanguageSpans.
  """
    languageSpans = _messages.MessageField('NlpSaftLanguageSpan', 1, repeated=True)
    probability = _messages.FloatField(2, variant=_messages.Variant.FLOAT)