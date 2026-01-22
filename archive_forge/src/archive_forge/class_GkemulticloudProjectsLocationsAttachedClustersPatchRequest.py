from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsAttachedClustersPatchRequest(_messages.Message):
    """A GkemulticloudProjectsLocationsAttachedClustersPatchRequest object.

  Fields:
    googleCloudGkemulticloudV1AttachedCluster: A
      GoogleCloudGkemulticloudV1AttachedCluster resource to be passed as the
      request body.
    name: The name of this resource. Cluster names are formatted as
      `projects//locations//attachedClusters/`. See [Resource
      Names](https://cloud.google.com/apis/design/resource_names) for more
      details on Google Cloud Platform resource names.
    updateMask: Required. Mask of fields to update. At least one path must be
      supplied in this field. The elements of the repeated paths field can
      only include these fields from AttachedCluster: * `annotations`. *
      `authorization.admin_groups`. * `authorization.admin_users`. *
      `binary_authorization.evaluation_mode`. * `description`. *
      `logging_config.component_config.enable_components`. *
      `monitoring_config.managed_prometheus_config.enabled`. *
      `platform_version`. * `proxy_config.kubernetes_secret.name`. *
      `proxy_config.kubernetes_secret.namespace`.
    validateOnly: If set, only validate the request, but do not actually
      update the cluster.
  """
    googleCloudGkemulticloudV1AttachedCluster = _messages.MessageField('GoogleCloudGkemulticloudV1AttachedCluster', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)