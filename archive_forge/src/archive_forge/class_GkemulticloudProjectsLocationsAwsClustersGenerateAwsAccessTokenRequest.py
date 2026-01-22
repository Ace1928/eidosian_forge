from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsAwsClustersGenerateAwsAccessTokenRequest(_messages.Message):
    """A GkemulticloudProjectsLocationsAwsClustersGenerateAwsAccessTokenRequest
  object.

  Fields:
    awsCluster: Required. The name of the AwsCluster resource to authenticate
      to. `AwsCluster` names are formatted as
      `projects//locations//awsClusters/`. See [Resource
      Names](https://cloud.google.com/apis/design/resource_names) for more
      details on Google Cloud resource names.
  """
    awsCluster = _messages.StringField(1, required=True)