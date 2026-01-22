from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeInstanceRestoreProperties(_messages.Message):
    """ComputeInstanceRestoreProperties represents Compute Engine instance
  properties to be overridden during restore.

  Enums:
    KeyRevocationActionTypeValueValuesEnum: Optional. KeyRevocationActionType
      of the instance.
    PrivateIpv6GoogleAccessValueValuesEnum: Optional. The private IPv6 google
      access type for the VM. If not specified, use INHERIT_FROM_SUBNETWORK as
      default.

  Messages:
    LabelsValue: Optional. Labels to apply to this instance.

  Fields:
    advancedMachineFeatures: Optional. Controls for advanced machine-related
      behavior features.
    canIpForward: Optional. Allows this instance to send and receive packets
      with non-matching destination or source IPs.
    confidentialInstanceConfig: Optional. Controls Confidential compute
      options on the instance
    deletionProtection: Optional. Whether the resource should be protected
      against deletion.
    description: Optional. An optional description of this resource. Provide
      this property when you create the resource.
    disks: Optional. Array of disks associated with this instance. Persistent
      disks must be created before you can assign them.
    displayDevice: Optional. Enables display device for the instance.
    guestAccelerators: Optional. A list of the type and count of accelerator
      cards attached to the instance.
    hostname: Optional. Specifies the hostname of the instance. The specified
      hostname must be RFC1035 compliant. If hostname is not specified, the
      default hostname is [INSTANCE_NAME].c.[PROJECT_ID].internal when using
      the global DNS, and [INSTANCE_NAME].[ZONE].c.[PROJECT_ID].internal when
      using zonal DNS.
    instanceEncryptionKey: Optional. Encrypts suspended data for an instance
      with a customer-managed encryption key.
    keyRevocationActionType: Optional. KeyRevocationActionType of the
      instance.
    labels: Optional. Labels to apply to this instance.
    machineType: Optional. Full or partial URL of the machine type resource to
      use for this instance.
    metadata: Optional. This includes custom metadata and predefined keys.
    minCpuPlatform: Optional. Minimum CPU platform to use for this instance.
    name: Required. Name of the compute instance.
    networkInterfaces: Optional. An array of network configurations for this
      instance. These specify how interfaces are configured to interact with
      other network services, such as connecting to the internet. Multiple
      interfaces are supported per instance.
    networkPerformanceConfig: Optional. Configure network performance such as
      egress bandwidth tier.
    params: Input only. Additional params passed with the request, but not
      persisted as part of resource payload.
    privateIpv6GoogleAccess: Optional. The private IPv6 google access type for
      the VM. If not specified, use INHERIT_FROM_SUBNETWORK as default.
    reservationAffinity: Optional. Specifies the reservations that this
      instance can consume from.
    resourcePolicies: Optional. Resource policies applied to this instance.
    scheduling: Optional. Sets the scheduling options for this instance.
    serviceAccounts: Optional. A list of service accounts, with their
      specified scopes, authorized for this instance. Only one service account
      per VM instance is supported.
    shieldedInstanceConfig: Optional. Controls Shielded compute options on the
      instance.
    tags: Optional. Tags to apply to this instance. Tags are used to identify
      valid sources or targets for network firewalls and are specified by the
      client during instance creation.
  """

    class KeyRevocationActionTypeValueValuesEnum(_messages.Enum):
        """Optional. KeyRevocationActionType of the instance.

    Values:
      KEY_REVOCATION_ACTION_TYPE_UNSPECIFIED: Default value. This value is
        unused.
      NONE: Indicates user chose no operation.
      STOP: Indicates user chose to opt for VM shutdown on key revocation.
    """
        KEY_REVOCATION_ACTION_TYPE_UNSPECIFIED = 0
        NONE = 1
        STOP = 2

    class PrivateIpv6GoogleAccessValueValuesEnum(_messages.Enum):
        """Optional. The private IPv6 google access type for the VM. If not
    specified, use INHERIT_FROM_SUBNETWORK as default.

    Values:
      INSTANCE_PRIVATE_IPV6_GOOGLE_ACCESS_UNSPECIFIED: Default value. This
        value is unused.
      INHERIT_FROM_SUBNETWORK: Each network interface inherits
        PrivateIpv6GoogleAccess from its subnetwork.
      ENABLE_OUTBOUND_VM_ACCESS_TO_GOOGLE: Outbound private IPv6 access from
        VMs in this subnet to Google services. If specified, the subnetwork
        who is attached to the instance's default network interface will be
        assigned an internal IPv6 prefix if it doesn't have before.
      ENABLE_BIDIRECTIONAL_ACCESS_TO_GOOGLE: Bidirectional private IPv6 access
        to/from Google services. If specified, the subnetwork who is attached
        to the instance's default network interface will be assigned an
        internal IPv6 prefix if it doesn't have before.
    """
        INSTANCE_PRIVATE_IPV6_GOOGLE_ACCESS_UNSPECIFIED = 0
        INHERIT_FROM_SUBNETWORK = 1
        ENABLE_OUTBOUND_VM_ACCESS_TO_GOOGLE = 2
        ENABLE_BIDIRECTIONAL_ACCESS_TO_GOOGLE = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Labels to apply to this instance.

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
    advancedMachineFeatures = _messages.MessageField('AdvancedMachineFeatures', 1)
    canIpForward = _messages.BooleanField(2)
    confidentialInstanceConfig = _messages.MessageField('ConfidentialInstanceConfig', 3)
    deletionProtection = _messages.BooleanField(4)
    description = _messages.StringField(5)
    disks = _messages.MessageField('AttachedDisk', 6, repeated=True)
    displayDevice = _messages.MessageField('DisplayDevice', 7)
    guestAccelerators = _messages.MessageField('AcceleratorConfig', 8, repeated=True)
    hostname = _messages.StringField(9)
    instanceEncryptionKey = _messages.MessageField('CustomerEncryptionKey', 10)
    keyRevocationActionType = _messages.EnumField('KeyRevocationActionTypeValueValuesEnum', 11)
    labels = _messages.MessageField('LabelsValue', 12)
    machineType = _messages.StringField(13)
    metadata = _messages.MessageField('Metadata', 14)
    minCpuPlatform = _messages.StringField(15)
    name = _messages.StringField(16)
    networkInterfaces = _messages.MessageField('NetworkInterface', 17, repeated=True)
    networkPerformanceConfig = _messages.MessageField('NetworkPerformanceConfig', 18)
    params = _messages.MessageField('InstanceParams', 19)
    privateIpv6GoogleAccess = _messages.EnumField('PrivateIpv6GoogleAccessValueValuesEnum', 20)
    reservationAffinity = _messages.MessageField('AllocationAffinity', 21)
    resourcePolicies = _messages.StringField(22, repeated=True)
    scheduling = _messages.MessageField('Scheduling', 23)
    serviceAccounts = _messages.MessageField('ServiceAccount', 24, repeated=True)
    shieldedInstanceConfig = _messages.MessageField('ShieldedInstanceConfig', 25)
    tags = _messages.MessageField('Tags', 26)