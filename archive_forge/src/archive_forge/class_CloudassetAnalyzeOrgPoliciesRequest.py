from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudassetAnalyzeOrgPoliciesRequest(_messages.Message):
    """A CloudassetAnalyzeOrgPoliciesRequest object.

  Fields:
    constraint: Required. The name of the constraint to analyze organization
      policies for. The response only contains analyzed organization policies
      for the provided constraint.
    filter: The expression to filter
      AnalyzeOrgPoliciesResponse.org_policy_results. Filtering is currently
      available for bare literal values and the following fields: *
      consolidated_policy.attached_resource *
      consolidated_policy.rules.enforce When filtering by a specific field,
      the only supported operator is `=`. For example, filtering by consolidat
      ed_policy.attached_resource="//cloudresourcemanager.googleapis.com/folde
      rs/001" will return all the Organization Policy results attached to
      "folders/001".
    pageSize: The maximum number of items to return per page. If unspecified,
      AnalyzeOrgPoliciesResponse.org_policy_results will contain 20 items with
      a maximum of 200.
    pageToken: The pagination token to retrieve the next page.
    scope: Required. The organization to scope the request. Only organization
      policies within the scope will be analyzed. *
      organizations/{ORGANIZATION_NUMBER} (e.g., "organizations/123456")
  """
    constraint = _messages.StringField(1)
    filter = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    scope = _messages.StringField(5, required=True)