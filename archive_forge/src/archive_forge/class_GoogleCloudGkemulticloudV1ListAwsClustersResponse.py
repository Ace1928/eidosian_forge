from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1ListAwsClustersResponse(_messages.Message):
    """Response message for `AwsClusters.ListAwsClusters` method.

  Fields:
    awsClusters: A list of AwsCluster resources in the specified Google Cloud
      Platform project and region region.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    awsClusters = _messages.MessageField('GoogleCloudGkemulticloudV1AwsCluster', 1, repeated=True)
    nextPageToken = _messages.StringField(2)