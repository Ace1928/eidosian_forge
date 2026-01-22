from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SddcProjectsLocationsClusterGroupsClustersAddNodesRequest(_messages.Message):
    """A SddcProjectsLocationsClusterGroupsClustersAddNodesRequest object.

  Fields:
    addNodesRequest: A AddNodesRequest resource to be passed as the request
      body.
    cluster: Required. The resource name of the `Cluster` to perform add
      nodes. Resource names are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. For example,
      projects/ PROJECT-NUMBER/locations/us-
      central1/clusterGroups/MY_GROUP/clusters/ MY_CLUSTER
  """
    addNodesRequest = _messages.MessageField('AddNodesRequest', 1)
    cluster = _messages.StringField(2, required=True)