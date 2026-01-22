from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2Proximity(_messages.Message):
    """Message for specifying a window around a finding to apply a detection
  rule.

  Fields:
    windowAfter: Number of characters after the finding to consider.
    windowBefore: Number of characters before the finding to consider. For
      tabular data, if you want to modify the likelihood of an entire column
      of findngs, set this to 1. For more information, see [Hotword example:
      Set the match likelihood of a table column]
      (https://cloud.google.com/sensitive-data-protection/docs/creating-
      custom-infotypes-likelihood#match-column-values).
  """
    windowAfter = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    windowBefore = _messages.IntegerField(2, variant=_messages.Variant.INT32)