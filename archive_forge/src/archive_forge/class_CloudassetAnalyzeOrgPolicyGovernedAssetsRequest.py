from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudassetAnalyzeOrgPolicyGovernedAssetsRequest(_messages.Message):
    """A CloudassetAnalyzeOrgPolicyGovernedAssetsRequest object.

  Fields:
    constraint: Required. The name of the constraint to analyze governed
      assets for. The analysis only contains analyzed organization policies
      for the provided constraint.
    filter: The expression to filter
      AnalyzeOrgPolicyGovernedAssetsResponse.governed_assets. For governed
      resources, filtering is currently available for bare literal values and
      the following fields: * governed_resource.project *
      governed_resource.folders * consolidated_policy.rules.enforce When
      filtering by `governed_resource.project` or
      `consolidated_policy.rules.enforce`, the only supported operator is `=`.
      When filtering by `governed_resource.folders`, the supported operators
      are `=` and `:`. For example, filtering by
      `governed_resource.project="projects/12345678"` will return all the
      governed resources under "projects/12345678", including the project
      itself if applicable. For governed IAM policies, filtering is currently
      available for bare literal values and the following fields: *
      governed_iam_policy.project * governed_iam_policy.folders *
      consolidated_policy.rules.enforce When filtering by
      `governed_iam_policy.project` or `consolidated_policy.rules.enforce`,
      the only supported operator is `=`. When filtering by
      `governed_iam_policy.folders`, the supported operators are `=` and `:`.
      For example, filtering by
      `governed_iam_policy.folders:"folders/12345678"` will return all the
      governed IAM policies under "folders/001".
    pageSize: The maximum number of items to return per page. If unspecified,
      AnalyzeOrgPolicyGovernedAssetsResponse.governed_assets will contain 100
      items with a maximum of 200.
    pageToken: The pagination token to retrieve the next page.
    scope: Required. The organization to scope the request. Only organization
      policies within the scope will be analyzed. The output assets will also
      be limited to the ones governed by those in-scope organization policies.
      * organizations/{ORGANIZATION_NUMBER} (e.g., "organizations/123456")
  """
    constraint = _messages.StringField(1)
    filter = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    scope = _messages.StringField(5, required=True)