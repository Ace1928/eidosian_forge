from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsNetworkPoliciesExternalAccessRulesListRequest(_messages.Message):
    """A
  VmwareengineProjectsLocationsNetworkPoliciesExternalAccessRulesListRequest
  object.

  Fields:
    filter: A filter expression that matches resources returned in the
      response. The expression must specify the field name, a comparison
      operator, and the value that you want to use for filtering. The value
      must be a string, a number, or a boolean. The comparison operator must
      be `=`, `!=`, `>`, or `<`. For example, if you are filtering a list of
      external access rules, you can exclude the ones named `example-rule` by
      specifying `name != "example-rule"`. To filter on multiple expressions,
      provide each separate expression within parentheses. For example: ```
      (name = "example-rule") (createTime > "2021-04-12T08:15:10.40Z") ``` By
      default, each expression is an `AND` expression. However, you can
      include `AND` and `OR` expressions explicitly. For example: ``` (name =
      "example-rule-1") AND (createTime > "2021-04-12T08:15:10.40Z") OR (name
      = "example-rule-2") ```
    orderBy: Sorts list results by a certain order. By default, returned
      results are ordered by `name` in ascending order. You can also sort
      results in descending order based on the `name` value using
      `orderBy="name desc"`. Currently, only ordering by `name` is supported.
    pageSize: The maximum number of external access rules to return in one
      page. The service may return fewer than this value. The maximum value is
      coerced to 1000. The default value of this field is 500.
    pageToken: A page token, received from a previous
      `ListExternalAccessRulesRequest` call. Provide this to retrieve the
      subsequent page. When paginating, all other parameters provided to
      `ListExternalAccessRulesRequest` must match the call that provided the
      page token.
    parent: Required. The resource name of the network policy to query for
      external access firewall rules. Resource names are schemeless URIs that
      follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/us-central1/networkPolicies/my-policy`
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)