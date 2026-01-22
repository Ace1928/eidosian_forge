from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterIamV3DenyRuleExplanationAnnotatedPermissionMatching(_messages.Message):
    """Details about whether the permission in the request is denied by the
  deny rule.

  Enums:
    PermissionMatchingStateValueValuesEnum: Indicates whether the permission
      in the request is denied by the deny rule.
    RelevanceValueValuesEnum: The relevance of the permission status to the
      overall determination for the rule.

  Fields:
    permissionMatchingState: Indicates whether the permission in the request
      is denied by the deny rule.
    relevance: The relevance of the permission status to the overall
      determination for the rule.
  """

    class PermissionMatchingStateValueValuesEnum(_messages.Enum):
        """Indicates whether the permission in the request is denied by the deny
    rule.

    Values:
      PERMISSION_PATTERN_MATCHING_STATE_UNSPECIFIED: Not specified.
      PERMISSION_PATTERN_MATCHED: The permission in the request matches the
        permission in the policy.
      PERMISSION_PATTERN_NOT_MATCHED: The permission in the request matches
        the permission in the policy.
    """
        PERMISSION_PATTERN_MATCHING_STATE_UNSPECIFIED = 0
        PERMISSION_PATTERN_MATCHED = 1
        PERMISSION_PATTERN_NOT_MATCHED = 2

    class RelevanceValueValuesEnum(_messages.Enum):
        """The relevance of the permission status to the overall determination
    for the rule.

    Values:
      HEURISTIC_RELEVANCE_UNSPECIFIED: Not specified.
      HEURISTIC_RELEVANCE_NORMAL: The data point has a limited effect on the
        result. Changing the data point is unlikely to affect the overall
        determination.
      HEURISTIC_RELEVANCE_HIGH: The data point has a strong effect on the
        result. Changing the data point is likely to affect the overall
        determination.
    """
        HEURISTIC_RELEVANCE_UNSPECIFIED = 0
        HEURISTIC_RELEVANCE_NORMAL = 1
        HEURISTIC_RELEVANCE_HIGH = 2
    permissionMatchingState = _messages.EnumField('PermissionMatchingStateValueValuesEnum', 1)
    relevance = _messages.EnumField('RelevanceValueValuesEnum', 2)