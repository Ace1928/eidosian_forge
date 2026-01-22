from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DetectionRule(_messages.Message):
    """Deprecated; use `InspectionRuleSet` instead. Rule for modifying a
  `CustomInfoType` to alter behavior under certain circumstances, depending on
  the specific details of the rule. Not supported for the `surrogate_type`
  custom infoType.

  Fields:
    hotwordRule: Hotword-based detection rule.
  """
    hotwordRule = _messages.MessageField('GooglePrivacyDlpV2HotwordRule', 1)