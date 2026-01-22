from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1ListAttachedClustersResponse(_messages.Message):
    """Response message for `AttachedClusters.ListAttachedClusters` method.

  Fields:
    attachedClusters: A list of AttachedCluster resources in the specified
      Google Cloud Platform project and region region.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    attachedClusters = _messages.MessageField('GoogleCloudGkemulticloudV1AttachedCluster', 1, repeated=True)
    nextPageToken = _messages.StringField(2)