from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsPrivateConnectionsListRequest(_messages.Message):
    """A VmwareengineProjectsLocationsPrivateConnectionsListRequest object.

  Fields:
    filter: A filter expression that matches resources returned in the
      response. The expression must specify the field name, a comparison
      operator, and the value that you want to use for filtering. The value
      must be a string, a number, or a boolean. The comparison operator must
      be `=`, `!=`, `>`, or `<`. For example, if you are filtering a list of
      private connections, you can exclude the ones named `example-connection`
      by specifying `name != "example-connection"`. To filter on multiple
      expressions, provide each separate expression within parentheses. For
      example: ``` (name = "example-connection") (createTime >
      "2022-09-22T08:15:10.40Z") ``` By default, each expression is an `AND`
      expression. However, you can include `AND` and `OR` expressions
      explicitly. For example: ``` (name = "example-connection-1") AND
      (createTime > "2021-04-12T08:15:10.40Z") OR (name = "example-
      connection-2") ```
    orderBy: Sorts list results by a certain order. By default, returned
      results are ordered by `name` in ascending order. You can also sort
      results in descending order based on the `name` value using
      `orderBy="name desc"`. Currently, only ordering by `name` is supported.
    pageSize: The maximum number of private connections to return in one page.
      The maximum value is coerced to 1000. The default value of this field is
      500.
    pageToken: A page token, received from a previous `ListPrivateConnections`
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to `ListPrivateConnections` must match the
      call that provided the page token.
    parent: Required. The resource name of the location to query for private
      connections. Resource names are schemeless URIs that follow the
      conventions in https://cloud.google.com/apis/design/resource_names. For
      example: `projects/my-project/locations/us-central1`
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)