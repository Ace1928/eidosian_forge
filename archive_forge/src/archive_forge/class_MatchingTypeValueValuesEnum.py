from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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