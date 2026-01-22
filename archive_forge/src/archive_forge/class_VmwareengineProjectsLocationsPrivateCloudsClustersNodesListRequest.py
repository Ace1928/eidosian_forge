from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsPrivateCloudsClustersNodesListRequest(_messages.Message):
    """A VmwareengineProjectsLocationsPrivateCloudsClustersNodesListRequest
  object.

  Fields:
    pageSize: The maximum number of nodes to return in one page. The service
      may return fewer than this value. The maximum value is coerced to 1000.
      The default value of this field is 500.
    pageToken: A page token, received from a previous `ListNodes` call.
      Provide this to retrieve the subsequent page. When paginating, all other
      parameters provided to `ListNodes` must match the call that provided the
      page token.
    parent: Required. The resource name of the cluster to be queried for
      nodes. Resource names are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/us-central1-a/privateClouds/my-
      cloud/clusters/my-cluster`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)