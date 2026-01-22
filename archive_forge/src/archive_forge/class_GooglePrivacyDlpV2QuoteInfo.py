from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2QuoteInfo(_messages.Message):
    """Message for infoType-dependent details parsed from quote.

  Fields:
    dateTime: The date time indicated by the quote.
  """
    dateTime = _messages.MessageField('GooglePrivacyDlpV2DateTime', 1)