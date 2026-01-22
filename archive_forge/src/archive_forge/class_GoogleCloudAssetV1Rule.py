from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssetV1Rule(_messages.Message):
    """This rule message is a customized version of the one defined in the
  Organization Policy system. In addition to the fields defined in the
  original organization policy, it contains additional field(s) under specific
  circumstances to support analysis results.

  Fields:
    allowAll: Setting this to true means that all values are allowed. This
      field can be set only in Policies for list constraints.
    condition: The evaluating condition for this rule.
    conditionEvaluation: The condition evaluation result for this rule. Only
      populated if it meets all the following criteria: * There is a condition
      defined for this rule. * This rule is within AnalyzeOrgPolicyGovernedCon
      tainersResponse.GovernedContainer.consolidated_policy, or
      AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset.consolidated_policy
      when the AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset has
      AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset.governed_resource.
    denyAll: Setting this to true means that all values are denied. This field
      can be set only in Policies for list constraints.
    enforce: If `true`, then the `Policy` is enforced. If `false`, then any
      configuration is acceptable. This field can be set only in Policies for
      boolean constraints.
    values: List of values to be used for this policy rule. This field can be
      set only in policies for list constraints.
  """
    allowAll = _messages.BooleanField(1)
    condition = _messages.MessageField('Expr', 2)
    conditionEvaluation = _messages.MessageField('ConditionEvaluation', 3)
    denyAll = _messages.BooleanField(4)
    enforce = _messages.BooleanField(5)
    values = _messages.MessageField('GoogleCloudAssetV1StringValues', 6)