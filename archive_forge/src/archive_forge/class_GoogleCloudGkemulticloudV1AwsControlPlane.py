from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AwsControlPlane(_messages.Message):
    """ControlPlane defines common parameters between control plane nodes.

  Messages:
    TagsValue: Optional. A set of AWS resource tags to propagate to all
      underlying managed AWS resources. Specify at most 50 pairs containing
      alphanumerics, spaces, and symbols (.+-=_:@/). Keys can be up to 127
      Unicode characters. Values can be up to 255 Unicode characters.

  Fields:
    awsServicesAuthentication: Required. Authentication configuration for
      management of AWS resources.
    configEncryption: Required. Config encryption for user data.
    databaseEncryption: Required. The ARN of the AWS KMS key used to encrypt
      cluster secrets.
    iamInstanceProfile: Required. The name or ARN of the AWS IAM instance
      profile to assign to each control plane replica.
    instancePlacement: Optional. The placement to use on control plane
      instances. When unspecified, the VPC's default tenancy will be used.
    instanceType: Optional. The AWS instance type. When unspecified, it uses a
      default based on the cluster's version.
    mainVolume: Optional. Configuration related to the main volume provisioned
      for each control plane replica. The main volume is in charge of storing
      all of the cluster's etcd state. Volumes will be provisioned in the
      availability zone associated with the corresponding subnet. When
      unspecified, it defaults to 8 GiB with the GP2 volume type.
    proxyConfig: Optional. Proxy configuration for outbound HTTP(S) traffic.
    rootVolume: Optional. Configuration related to the root volume provisioned
      for each control plane replica. Volumes will be provisioned in the
      availability zone associated with the corresponding subnet. When
      unspecified, it defaults to 32 GiB with the GP2 volume type.
    securityGroupIds: Optional. The IDs of additional security groups to add
      to control plane replicas. The Anthos Multi-Cloud API will automatically
      create and manage security groups with the minimum rules needed for a
      functioning cluster.
    sshConfig: Optional. SSH configuration for how to access the underlying
      control plane machines.
    subnetIds: Required. The list of subnets where control plane replicas will
      run. A replica will be provisioned on each subnet and up to three values
      can be provided. Each subnet must be in a different AWS Availability
      Zone (AZ).
    tags: Optional. A set of AWS resource tags to propagate to all underlying
      managed AWS resources. Specify at most 50 pairs containing
      alphanumerics, spaces, and symbols (.+-=_:@/). Keys can be up to 127
      Unicode characters. Values can be up to 255 Unicode characters.
    version: Required. The Kubernetes version to run on control plane replicas
      (e.g. `1.19.10-gke.1000`). You can list all supported versions on a
      given Google Cloud region by calling GetAwsServerConfig.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class TagsValue(_messages.Message):
        """Optional. A set of AWS resource tags to propagate to all underlying
    managed AWS resources. Specify at most 50 pairs containing alphanumerics,
    spaces, and symbols (.+-=_:@/). Keys can be up to 127 Unicode characters.
    Values can be up to 255 Unicode characters.

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
    awsServicesAuthentication = _messages.MessageField('GoogleCloudGkemulticloudV1AwsServicesAuthentication', 1)
    configEncryption = _messages.MessageField('GoogleCloudGkemulticloudV1AwsConfigEncryption', 2)
    databaseEncryption = _messages.MessageField('GoogleCloudGkemulticloudV1AwsDatabaseEncryption', 3)
    iamInstanceProfile = _messages.StringField(4)
    instancePlacement = _messages.MessageField('GoogleCloudGkemulticloudV1AwsInstancePlacement', 5)
    instanceType = _messages.StringField(6)
    mainVolume = _messages.MessageField('GoogleCloudGkemulticloudV1AwsVolumeTemplate', 7)
    proxyConfig = _messages.MessageField('GoogleCloudGkemulticloudV1AwsProxyConfig', 8)
    rootVolume = _messages.MessageField('GoogleCloudGkemulticloudV1AwsVolumeTemplate', 9)
    securityGroupIds = _messages.StringField(10, repeated=True)
    sshConfig = _messages.MessageField('GoogleCloudGkemulticloudV1AwsSshConfig', 11)
    subnetIds = _messages.StringField(12, repeated=True)
    tags = _messages.MessageField('TagsValue', 13)
    version = _messages.StringField(14)