from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsPrivateCloudsClustersListRequest(_messages.Message):
    """A VmwareengineProjectsLocationsPrivateCloudsClustersListRequest object.

  Fields:
    filter:  To filter on multiple expressions, provide each separate
      expression within parentheses. For example: ``` (name = "example-
      cluster") (nodeCount = "3") ``` By default, each expression is an `AND`
      expression. However, you can include `AND` and `OR` expressions
      explicitly. For example: ``` (name = "example-cluster-1") AND
      (createTime > "2021-04-12T08:15:10.40Z") OR (name = "example-cluster-2")
      ```
    orderBy: Sorts list results by a certain order. By default, returned
      results are ordered by `name` in ascending order. You can also sort
      results in descending order based on the `name` value using
      `orderBy="name desc"`. Currently, only ordering by `name` is supported.
    pageSize: The maximum number of clusters to return in one page. The
      service may return fewer than this value. The maximum value is coerced
      to 1000. The default value of this field is 500.
    pageToken: A page token, received from a previous `ListClusters` call.
      Provide this to retrieve the subsequent page. When paginating, all other
      parameters provided to `ListClusters` must match the call that provided
      the page token.
    parent: Required. The resource name of the private cloud to query for
      clusters. Resource names are schemeless URIs that follow the conventions
      in https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/us-central1-a/privateClouds/my-cloud`
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)