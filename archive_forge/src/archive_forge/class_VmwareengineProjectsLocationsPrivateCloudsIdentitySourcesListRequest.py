from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsPrivateCloudsIdentitySourcesListRequest(_messages.Message):
    """A VmwareengineProjectsLocationsPrivateCloudsIdentitySourcesListRequest
  object.

  Fields:
    filter: Optional. A filter expression that matches resources returned in
      the response. The expression must specify the field name, a comparison
      operator, and the value that you want to use for filtering. The value
      must be a string, a number, or a boolean. The comparison operator must
      be `=`, `!=`, `>`, `<` or `:`. For example, if you are filtering a list
      of identity sources, you can exclude the ones named `example-identity-
      source` by specifying `name != "example-identity-source"`. Wildcard
      filters are partially supported. Only operator _has_ (`:`) is available.
      For example, you can select all resources with `name` field starting
      with "foo" by specifying the following filter: `name : "foo*"`. To
      filter on multiple expressions, provide each separate expression within
      parentheses. For example: ``` (name = "example-identity-source")
      (domain_name = "example-domain-name") ``` By default, each expression is
      an `AND` expression. However, you can include `AND` and `OR` expressions
      explicitly. For example: ``` (name = "example-identity-source-1") AND
      (createTime > "2021-04-12T08:15:10.40Z") OR (name = "example-identity-
      source-2") ```
    orderBy: Optional. Sorts list results by a certain order. By default,
      returned results are ordered by `name` in ascending order. You can also
      sort results in descending order based on the `name` value using
      `orderBy="name desc"`. Currently, only ordering by `name` is supported.
      In case of cross-parent requests, correct ordering is not guaranteed.
    pageSize: Optional. The maximum number of identity sources to return in
      one page. The service may return fewer than this value. The maximum
      value is coerced to 1000. The default value of this field is 500.
    pageToken: Optional. A page token, received from a previous
      `ListIdentitySources` call. Provide this to retrieve the subsequent
      page. When paginating, all other parameters provided to
      `ListIdentitySources` must match the call that provided the page token.
    parent: Required. The resource name of the private cloud to query for
      identity sources. Resource names are schemeless URIs that follow the
      conventions in https://cloud.google.com/apis/design/resource_names, for
      example: `projects/my-project/locations/us-central1-a/privateClouds/my-
      cloud`. Parent resource name can contain wildcards, for example:
      `projects/my-project/locations/-/privateClouds/my-cloud`.
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)