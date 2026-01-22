from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceConfig(_messages.Message):
    """Configuration parameters for a new instance.

  Enums:
    NetworkConfigValueValuesEnum: The type of network configuration on the
      instance.

  Fields:
    accountNetworksEnabled: If true networks can be from different projects of
      the same vendor account.
    clientNetwork: Client network address. Filled if
      InstanceConfig.multivlan_config is false.
    hyperthreading: Whether the instance should be provisioned with
      Hyperthreading enabled.
    id: A transient unique identifier to idenfity an instance within an
      ProvisioningConfig request.
    instanceType: Instance type. [Available
      types](https://cloud.google.com/bare-metal/docs/bms-
      planning#server_configurations)
    kmsKeyVersion: Name of the KMS crypto key version used to encrypt the
      initial passwords. The key has to have ASYMMETRIC_DECRYPT purpose.
    logicalInterfaces: List of logical interfaces for the instance. The number
      of logical interfaces will be the same as number of hardware bond/nic on
      the chosen network template. Filled if InstanceConfig.multivlan_config
      is true.
    name: The name of the instance config.
    networkConfig: The type of network configuration on the instance.
    networkTemplate: Server network template name. Filled if
      InstanceConfig.multivlan_config is true.
    osImage: OS image to initialize the instance. [Available
      images](https://cloud.google.com/bare-metal/docs/bms-
      planning#server_configurations)
    privateNetwork: Private network address, if any. Filled if
      InstanceConfig.multivlan_config is false.
    sshKeyNames: Optional. List of names of ssh keys used to provision the
      instance.
    userNote: User note field, it can be used by customers to add additional
      information for the BMS Ops team .
  """

    class NetworkConfigValueValuesEnum(_messages.Enum):
        """The type of network configuration on the instance.

    Values:
      NETWORKCONFIG_UNSPECIFIED: The unspecified network configuration.
      SINGLE_VLAN: Instance part of single client network and single private
        network.
      MULTI_VLAN: Instance part of multiple (or single) client networks and
        private networks.
    """
        NETWORKCONFIG_UNSPECIFIED = 0
        SINGLE_VLAN = 1
        MULTI_VLAN = 2
    accountNetworksEnabled = _messages.BooleanField(1)
    clientNetwork = _messages.MessageField('NetworkAddress', 2)
    hyperthreading = _messages.BooleanField(3)
    id = _messages.StringField(4)
    instanceType = _messages.StringField(5)
    kmsKeyVersion = _messages.StringField(6)
    logicalInterfaces = _messages.MessageField('GoogleCloudBaremetalsolutionV2LogicalInterface', 7, repeated=True)
    name = _messages.StringField(8)
    networkConfig = _messages.EnumField('NetworkConfigValueValuesEnum', 9)
    networkTemplate = _messages.StringField(10)
    osImage = _messages.StringField(11)
    privateNetwork = _messages.MessageField('NetworkAddress', 12)
    sshKeyNames = _messages.StringField(13, repeated=True)
    userNote = _messages.StringField(14)