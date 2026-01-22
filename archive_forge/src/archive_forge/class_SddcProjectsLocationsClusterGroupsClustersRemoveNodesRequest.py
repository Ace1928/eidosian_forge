from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SddcProjectsLocationsClusterGroupsClustersRemoveNodesRequest(_messages.Message):
    """A SddcProjectsLocationsClusterGroupsClustersRemoveNodesRequest object.

  Fields:
    cluster: Required. The resource name of the `Cluster` to perform remove
      nodes. Resource names are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example,
      projects/ PROJECT-NUMBER/locations/us-
      central1/clusterGroups/MY_GROUP/clusters/ MY_CLUSTER
    removeNodesRequest: A RemoveNodesRequest resource to be passed as the
      request body.
  """
    cluster = _messages.StringField(1, required=True)
    removeNodesRequest = _messages.MessageField('RemoveNodesRequest', 2)