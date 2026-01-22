from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsAwsClustersCreateRequest(_messages.Message):
    """A GkemulticloudProjectsLocationsAwsClustersCreateRequest object.

  Fields:
    awsClusterId: Required. A client provided ID the resource. Must be unique
      within the parent resource. The provided ID will be part of the
      AwsCluster resource name formatted as
      `projects//locations//awsClusters/`. Valid characters are `/a-z-/`.
      Cannot be longer than 63 characters.
    googleCloudGkemulticloudV1AwsCluster: A
      GoogleCloudGkemulticloudV1AwsCluster resource to be passed as the
      request body.
    parent: Required. The parent location where this AwsCluster resource will
      be created. Location names are formatted as `projects//locations/`. See
      [Resource Names](https://cloud.google.com/apis/design/resource_names)
      for more details on Google Cloud resource names.
    validateOnly: If set, only validate the request, but do not actually
      create the cluster.
  """
    awsClusterId = _messages.StringField(1)
    googleCloudGkemulticloudV1AwsCluster = _messages.MessageField('GoogleCloudGkemulticloudV1AwsCluster', 2)
    parent = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)