from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NlpSaftLangIdLocalesResult(_messages.Message):
    """A NlpSaftLangIdLocalesResult object.

  Fields:
    predictions: List of locales in which the text would be considered
      acceptable. Sorted in descending order according to each locale's
      respective likelihood. For example, if a Portuguese text is acceptable
      in both Brazil and Portugal, but is more strongly associated with
      Brazil, then the predictions would be ["pt-BR", "pt-PT"], in that order.
      May be empty, indicating that the model did not predict any acceptable
      locales.
  """
    predictions = _messages.MessageField('NlpSaftLangIdLocalesResultLocale', 1, repeated=True)