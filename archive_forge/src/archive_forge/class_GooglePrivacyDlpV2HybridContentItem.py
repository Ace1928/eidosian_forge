from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2HybridContentItem(_messages.Message):
    """An individual hybrid item to inspect. Will be stored temporarily during
  processing.

  Fields:
    findingDetails: Supplementary information that will be added to each
      finding.
    item: The item to inspect.
  """
    findingDetails = _messages.MessageField('GooglePrivacyDlpV2HybridFindingDetails', 1)
    item = _messages.MessageField('GooglePrivacyDlpV2ContentItem', 2)