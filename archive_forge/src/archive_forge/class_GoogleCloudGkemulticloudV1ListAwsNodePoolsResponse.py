from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1ListAwsNodePoolsResponse(_messages.Message):
    """Response message for `AwsClusters.ListAwsNodePools` method.

  Fields:
    awsNodePools: A list of AwsNodePool resources in the specified
      `AwsCluster`.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    awsNodePools = _messages.MessageField('GoogleCloudGkemulticloudV1AwsNodePool', 1, repeated=True)
    nextPageToken = _messages.StringField(2)