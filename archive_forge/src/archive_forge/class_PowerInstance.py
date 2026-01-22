from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PowerInstance(_messages.Message):
    """A Power instance.

  Enums:
    StateValueValuesEnum: Output only. The state of the instance.
    VirtualCpuTypeValueValuesEnum: Required. The processor type of this
      instance.

  Fields:
    addresses: Output only. List of addresses associated with this instance,
      corresponds to `addresses` field from Power's API.
    bootImage: Required. The name of the boot image used to create this
      instance.
    createTime: Output only. Instance creation time.
    healthStatus: Output only. Last health status for instance.
    memoryGib: Required. Memory size for the instance.
    name: Identifier. The resource name of this PowerInstance. Resource names
      are schemeless URIs that follow the conventions in
      https://cloud.google.com/apis/design/resource_names. Format: `projects/{
      project}/locations/{location}/powerInstances/{power_instance}`
    networkAttachments: Optional. List of network attachments associated with
      this instance, corresponds to `networks` field from Power's API.
    osImage: Required. The OS image currently installed on this instance.
    state: Output only. The state of the instance.
    systemType: Required. IBM Power System type, most commonly s922.
    uid: Output only. An unique identifier generated for the PowerInstance.
    updateTime: Output only. Instance update time.
    virtualCpuCores: Required. Processor for the instance.
    virtualCpuType: Required. The processor type of this instance.
    volumeIds: Optional. List of volumes IDs associated with this instance.
    volumes: Output only. List of volumes associated with this instance,
      retrieved by calling ListVolumes API.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the instance.

    Values:
      STATE_UNSPECIFIED: The state of the instance is unknown.
      ACTIVE: The instance is ACTIVE.
      SHUTOFF: The instance has been SHUTOFF.
      INACTIVE: The instance is INACTIVE.
      TIMEOUT: The instance is TIMEOUT.
      BUILD: The instance is BUILDing.
      REBOOT: The instance is REBOOTing.
      WARNING: The instance is in WARNING status.
      ERROR: The instance has ERROR.
      RESIZE: The instance is resizing.
      VERIFY_RESIZE: The instance is verifying resize.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        SHUTOFF = 2
        INACTIVE = 3
        TIMEOUT = 4
        BUILD = 5
        REBOOT = 6
        WARNING = 7
        ERROR = 8
        RESIZE = 9
        VERIFY_RESIZE = 10

    class VirtualCpuTypeValueValuesEnum(_messages.Enum):
        """Required. The processor type of this instance.

    Values:
      VIRTUAL_CPU_TYPE_UNSPECIFIED: Unspecified.
      DEDICATED: Dedicated processors. Processor counts for this type must be
        whole numbers.
      UNCAPPED_SHARED: Uncapped shared processors.
      CAPPED_SHARED: Capped shared processors.
    """
        VIRTUAL_CPU_TYPE_UNSPECIFIED = 0
        DEDICATED = 1
        UNCAPPED_SHARED = 2
        CAPPED_SHARED = 3
    addresses = _messages.MessageField('NetworkAttachment', 1, repeated=True)
    bootImage = _messages.StringField(2)
    createTime = _messages.StringField(3)
    healthStatus = _messages.StringField(4)
    memoryGib = _messages.IntegerField(5)
    name = _messages.StringField(6)
    networkAttachments = _messages.MessageField('NetworkAttachment', 7, repeated=True)
    osImage = _messages.MessageField('OsImage', 8)
    state = _messages.EnumField('StateValueValuesEnum', 9)
    systemType = _messages.StringField(10)
    uid = _messages.StringField(11)
    updateTime = _messages.StringField(12)
    virtualCpuCores = _messages.FloatField(13)
    virtualCpuType = _messages.EnumField('VirtualCpuTypeValueValuesEnum', 14)
    volumeIds = _messages.StringField(15, repeated=True)
    volumes = _messages.MessageField('PowerVolume', 16, repeated=True)