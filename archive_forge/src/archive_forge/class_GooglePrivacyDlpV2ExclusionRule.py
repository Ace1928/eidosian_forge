from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2ExclusionRule(_messages.Message):
    """The rule that specifies conditions when findings of infoTypes specified
  in `InspectionRuleSet` are removed from results.

  Enums:
    MatchingTypeValueValuesEnum: How the rule is applied, see MatchingType
      documentation for details.

  Fields:
    dictionary: Dictionary which defines the rule.
    excludeByHotword: Drop if the hotword rule is contained in the proximate
      context. For tabular data, the context includes the column name.
    excludeInfoTypes: Set of infoTypes for which findings would affect this
      rule.
    matchingType: How the rule is applied, see MatchingType documentation for
      details.
    regex: Regular expression which defines the rule.
  """

    class MatchingTypeValueValuesEnum(_messages.Enum):
        """How the rule is applied, see MatchingType documentation for details.

    Values:
      MATCHING_TYPE_UNSPECIFIED: Invalid.
      MATCHING_TYPE_FULL_MATCH: Full match. - Dictionary: join of Dictionary
        results matched complete finding quote - Regex: all regex matches fill
        a finding quote start to end - Exclude info type: completely inside
        affecting info types findings
      MATCHING_TYPE_PARTIAL_MATCH: Partial match. - Dictionary: at least one
        of the tokens in the finding matches - Regex: substring of the finding
        matches - Exclude info type: intersects with affecting info types
        findings
      MATCHING_TYPE_INVERSE_MATCH: Inverse match. - Dictionary: no tokens in
        the finding match the dictionary - Regex: finding doesn't match the
        regex - Exclude info type: no intersection with affecting info types
        findings
    """
        MATCHING_TYPE_UNSPECIFIED = 0
        MATCHING_TYPE_FULL_MATCH = 1
        MATCHING_TYPE_PARTIAL_MATCH = 2
        MATCHING_TYPE_INVERSE_MATCH = 3
    dictionary = _messages.MessageField('GooglePrivacyDlpV2Dictionary', 1)
    excludeByHotword = _messages.MessageField('GooglePrivacyDlpV2ExcludeByHotword', 2)
    excludeInfoTypes = _messages.MessageField('GooglePrivacyDlpV2ExcludeInfoTypes', 3)
    matchingType = _messages.EnumField('MatchingTypeValueValuesEnum', 4)
    regex = _messages.MessageField('GooglePrivacyDlpV2Regex', 5)