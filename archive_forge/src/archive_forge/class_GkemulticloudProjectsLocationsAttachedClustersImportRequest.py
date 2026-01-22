from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsAttachedClustersImportRequest(_messages.Message):
    """A GkemulticloudProjectsLocationsAttachedClustersImportRequest object.

  Fields:
    googleCloudGkemulticloudV1ImportAttachedClusterRequest: A
      GoogleCloudGkemulticloudV1ImportAttachedClusterRequest resource to be
      passed as the request body.
    parent: Required. The parent location where this AttachedCluster resource
      will be created. Location names are formatted as `projects//locations/`.
      See [Resource
      Names](https://cloud.google.com/apis/design/resource_names) for more
      details on Google Cloud resource names.
  """
    googleCloudGkemulticloudV1ImportAttachedClusterRequest = _messages.MessageField('GoogleCloudGkemulticloudV1ImportAttachedClusterRequest', 1)
    parent = _messages.StringField(2, required=True)