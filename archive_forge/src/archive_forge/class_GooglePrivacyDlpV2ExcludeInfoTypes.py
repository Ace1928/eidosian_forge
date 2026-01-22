from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2ExcludeInfoTypes(_messages.Message):
    """List of excluded infoTypes.

  Fields:
    infoTypes: InfoType list in ExclusionRule rule drops a finding when it
      overlaps or contained within with a finding of an infoType from this
      list. For example, for `InspectionRuleSet.info_types` containing
      "PHONE_NUMBER"` and `exclusion_rule` containing
      `exclude_info_types.info_types` with "EMAIL_ADDRESS" the phone number
      findings are dropped if they overlap with EMAIL_ADDRESS finding. That
      leads to "555-222-2222@example.org" to generate only a single finding,
      namely email address.
  """
    infoTypes = _messages.MessageField('GooglePrivacyDlpV2InfoType', 1, repeated=True)