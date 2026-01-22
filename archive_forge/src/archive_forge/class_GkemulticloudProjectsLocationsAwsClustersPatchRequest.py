from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GkemulticloudProjectsLocationsAwsClustersPatchRequest(_messages.Message):
    """A GkemulticloudProjectsLocationsAwsClustersPatchRequest object.

  Fields:
    googleCloudGkemulticloudV1AwsCluster: A
      GoogleCloudGkemulticloudV1AwsCluster resource to be passed as the
      request body.
    name: The name of this resource. Cluster names are formatted as
      `projects//locations//awsClusters/`. See [Resource
      Names](https://cloud.google.com/apis/design/resource_names) for more
      details on Google Cloud Platform resource names.
    updateMask: Required. Mask of fields to update. At least one path must be
      supplied in this field. The elements of the repeated paths field can
      only include these fields from AwsCluster: * `description`. *
      `annotations`. * `control_plane.version`. * `authorization.admin_users`.
      * `authorization.admin_groups`. *
      `binary_authorization.evaluation_mode`. *
      `control_plane.aws_services_authentication.role_arn`. *
      `control_plane.aws_services_authentication.role_session_name`. *
      `control_plane.config_encryption.kms_key_arn`. *
      `control_plane.instance_type`. * `control_plane.security_group_ids`. *
      `control_plane.proxy_config`. * `control_plane.proxy_config.secret_arn`.
      * `control_plane.proxy_config.secret_version`. *
      `control_plane.root_volume.size_gib`. *
      `control_plane.root_volume.volume_type`. *
      `control_plane.root_volume.iops`. *
      `control_plane.root_volume.throughput`. *
      `control_plane.root_volume.kms_key_arn`. * `control_plane.ssh_config`. *
      `control_plane.ssh_config.ec2_key_pair`. *
      `control_plane.instance_placement.tenancy`. *
      `control_plane.iam_instance_profile`. *
      `logging_config.component_config.enable_components`. *
      `control_plane.tags`. *
      `monitoring_config.managed_prometheus_config.enabled`. *
      `networking.per_node_pool_sg_rules_disabled`.
    validateOnly: If set, only validate the request, but do not actually
      update the cluster.
  """
    googleCloudGkemulticloudV1AwsCluster = _messages.MessageField('GoogleCloudGkemulticloudV1AwsCluster', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)