from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2HotwordRule(_messages.Message):
    """The rule that adjusts the likelihood of findings within a certain
  proximity of hotwords.

  Fields:
    hotwordRegex: Regular expression pattern defining what qualifies as a
      hotword.
    likelihoodAdjustment: Likelihood adjustment to apply to all matching
      findings.
    proximity: Range of characters within which the entire hotword must
      reside. The total length of the window cannot exceed 1000 characters.
      The finding itself will be included in the window, so that hotwords can
      be used to match substrings of the finding itself. Suppose you want
      Cloud DLP to promote the likelihood of the phone number regex "\\(\\d{3}\\)
      \\d{3}-\\d{4}" if the area code is known to be the area code of a
      company's office. In this case, use the hotword regex "\\(xxx\\)", where
      "xxx" is the area code in question. For tabular data, if you want to
      modify the likelihood of an entire column of findngs, see [Hotword
      example: Set the match likelihood of a table column]
      (https://cloud.google.com/sensitive-data-protection/docs/creating-
      custom-infotypes-likelihood#match-column-values).
  """
    hotwordRegex = _messages.MessageField('GooglePrivacyDlpV2Regex', 1)
    likelihoodAdjustment = _messages.MessageField('GooglePrivacyDlpV2LikelihoodAdjustment', 2)
    proximity = _messages.MessageField('GooglePrivacyDlpV2Proximity', 3)