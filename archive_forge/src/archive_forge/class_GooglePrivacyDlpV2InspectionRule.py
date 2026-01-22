from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2InspectionRule(_messages.Message):
    """A single inspection rule to be applied to infoTypes, specified in
  `InspectionRuleSet`.

  Fields:
    exclusionRule: Exclusion rule.
    hotwordRule: Hotword-based detection rule.
  """
    exclusionRule = _messages.MessageField('GooglePrivacyDlpV2ExclusionRule', 1)
    hotwordRule = _messages.MessageField('GooglePrivacyDlpV2HotwordRule', 2)