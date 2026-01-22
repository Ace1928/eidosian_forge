from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssetV1AnalyzeOrgPolicyGovernedAssetsResponseGovernedAsset(_messages.Message):
    """Represents a Google Cloud asset(resource or IAM policy) governed by the
  organization policies of the
  AnalyzeOrgPolicyGovernedAssetsRequest.constraint.

  Fields:
    consolidatedPolicy: The consolidated policy for the analyzed asset. The
      consolidated policy is computed by merging and evaluating
      AnalyzeOrgPolicyGovernedAssetsResponse.GovernedAsset.policy_bundle. The
      evaluation will respect the organization policy [hierarchy
      rules](https://cloud.google.com/resource-manager/docs/organization-
      policy/understanding-hierarchy).
    governedIamPolicy: An IAM policy governed by the organization policies of
      the AnalyzeOrgPolicyGovernedAssetsRequest.constraint.
    governedResource: A Google Cloud resource governed by the organization
      policies of the AnalyzeOrgPolicyGovernedAssetsRequest.constraint.
    policyBundle: The ordered list of all organization policies from the Analy
      zeOrgPoliciesResponse.OrgPolicyResult.consolidated_policy.attached_resou
      rce to the scope specified in the request. If the constraint is defined
      with default policy, it will also appear in the list.
  """
    consolidatedPolicy = _messages.MessageField('AnalyzerOrgPolicy', 1)
    governedIamPolicy = _messages.MessageField('GoogleCloudAssetV1AnalyzeOrgPolicyGovernedAssetsResponseGovernedIamPolicy', 2)
    governedResource = _messages.MessageField('GoogleCloudAssetV1AnalyzeOrgPolicyGovernedAssetsResponseGovernedResource', 3)
    policyBundle = _messages.MessageField('AnalyzerOrgPolicy', 4, repeated=True)