from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MultiCloudCluster(_messages.Message):
    """MultiCloudCluster contains information specific to GKE Multi-Cloud
  clusters.

  Fields:
    clusterMissing: Output only. If cluster_missing is set then it denotes
      that API(gkemulticloud.googleapis.com) resource for this GKE Multi-Cloud
      cluster no longer exists.
    resourceLink: Immutable. Self-link of the Google Cloud resource for the
      GKE Multi-Cloud cluster. For example:
      //gkemulticloud.googleapis.com/projects/my-project/locations/us-
      west1-a/awsClusters/my-cluster
      //gkemulticloud.googleapis.com/projects/my-project/locations/us-
      west1-a/azureClusters/my-cluster
      //gkemulticloud.googleapis.com/projects/my-project/locations/us-
      west1-a/attachedClusters/my-cluster
  """
    clusterMissing = _messages.BooleanField(1)
    resourceLink = _messages.StringField(2)