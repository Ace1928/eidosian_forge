from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AwsNodeConfig(_messages.Message):
    """Parameters that describe the nodes in a cluster.

  Messages:
    LabelsValue: Optional. The initial labels assigned to nodes of this node
      pool. An object containing a list of "key": value pairs. Example: {
      "name": "wrench", "mass": "1.3kg", "count": "3" }.
    TagsValue: Optional. Key/value metadata to assign to each underlying AWS
      resource. Specify at most 50 pairs containing alphanumerics, spaces, and
      symbols (.+-=_:@/). Keys can be up to 127 Unicode characters. Values can
      be up to 255 Unicode characters.

  Fields:
    autoscalingMetricsCollection: Optional. Configuration related to
      CloudWatch metrics collection on the Auto Scaling group of the node
      pool. When unspecified, metrics collection is disabled.
    configEncryption: Required. Config encryption for user data.
    iamInstanceProfile: Required. The name or ARN of the AWS IAM instance
      profile to assign to nodes in the pool.
    imageType: Optional. The OS image type to use on node pool instances. Can
      be unspecified, or have a value of `ubuntu`. When unspecified, it
      defaults to `ubuntu`.
    instancePlacement: Optional. Placement related info for this node. When
      unspecified, the VPC's default tenancy will be used.
    instanceType: Optional. The EC2 instance type when creating on-Demand
      instances. If unspecified during node pool creation, a default will be
      chosen based on the node pool version, and assigned to this field.
    labels: Optional. The initial labels assigned to nodes of this node pool.
      An object containing a list of "key": value pairs. Example: { "name":
      "wrench", "mass": "1.3kg", "count": "3" }.
    proxyConfig: Optional. Proxy configuration for outbound HTTP(S) traffic.
    rootVolume: Optional. Template for the root volume provisioned for node
      pool nodes. Volumes will be provisioned in the availability zone
      assigned to the node pool subnet. When unspecified, it defaults to 32
      GiB with the GP2 volume type.
    securityGroupIds: Optional. The IDs of additional security groups to add
      to nodes in this pool. The manager will automatically create security
      groups with minimum rules needed for a functioning cluster.
    spotConfig: Optional. Configuration for provisioning EC2 Spot instances
      When specified, the node pool will provision Spot instances from the set
      of spot_config.instance_types. This field is mutually exclusive with
      `instance_type`.
    sshConfig: Optional. The SSH configuration.
    tags: Optional. Key/value metadata to assign to each underlying AWS
      resource. Specify at most 50 pairs containing alphanumerics, spaces, and
      symbols (.+-=_:@/). Keys can be up to 127 Unicode characters. Values can
      be up to 255 Unicode characters.
    taints: Optional. The initial taints assigned to nodes of this node pool.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. The initial labels assigned to nodes of this node pool. An
    object containing a list of "key": value pairs. Example: { "name":
    "wrench", "mass": "1.3kg", "count": "3" }.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class TagsValue(_messages.Message):
        """Optional. Key/value metadata to assign to each underlying AWS
    resource. Specify at most 50 pairs containing alphanumerics, spaces, and
    symbols (.+-=_:@/). Keys can be up to 127 Unicode characters. Values can
    be up to 255 Unicode characters.

    Messages:
      AdditionalProperty: An additional property for a TagsValue object.

    Fields:
      additionalProperties: Additional properties of type TagsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a TagsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    autoscalingMetricsCollection = _messages.MessageField('GoogleCloudGkemulticloudV1AwsAutoscalingGroupMetricsCollection', 1)
    configEncryption = _messages.MessageField('GoogleCloudGkemulticloudV1AwsConfigEncryption', 2)
    iamInstanceProfile = _messages.StringField(3)
    imageType = _messages.StringField(4)
    instancePlacement = _messages.MessageField('GoogleCloudGkemulticloudV1AwsInstancePlacement', 5)
    instanceType = _messages.StringField(6)
    labels = _messages.MessageField('LabelsValue', 7)
    proxyConfig = _messages.MessageField('GoogleCloudGkemulticloudV1AwsProxyConfig', 8)
    rootVolume = _messages.MessageField('GoogleCloudGkemulticloudV1AwsVolumeTemplate', 9)
    securityGroupIds = _messages.StringField(10, repeated=True)
    spotConfig = _messages.MessageField('GoogleCloudGkemulticloudV1SpotConfig', 11)
    sshConfig = _messages.MessageField('GoogleCloudGkemulticloudV1AwsSshConfig', 12)
    tags = _messages.MessageField('TagsValue', 13)
    taints = _messages.MessageField('GoogleCloudGkemulticloudV1NodeTaint', 14, repeated=True)