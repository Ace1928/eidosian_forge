from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2InspectionRuleSet(_messages.Message):
    """Rule set for modifying a set of infoTypes to alter behavior under
  certain circumstances, depending on the specific details of the rules within
  the set.

  Fields:
    infoTypes: List of infoTypes this rule set is applied to.
    rules: Set of rules to be applied to infoTypes. The rules are applied in
      order.
  """
    infoTypes = _messages.MessageField('GooglePrivacyDlpV2InfoType', 1, repeated=True)
    rules = _messages.MessageField('GooglePrivacyDlpV2InspectionRule', 2, repeated=True)