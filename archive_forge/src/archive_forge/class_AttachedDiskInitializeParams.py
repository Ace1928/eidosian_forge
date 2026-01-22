from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AttachedDiskInitializeParams(_messages.Message):
    """[Input Only] Specifies the parameters for a new disk that will be
  created alongside the new instance. Use initialization parameters to create
  boot disks or local SSDs attached to the new instance. This field is
  persisted and returned for instanceTemplate and not returned in the context
  of instance. This property is mutually exclusive with the source property;
  you can only define one or the other, but not both.

  Enums:
    ArchitectureValueValuesEnum: The architecture of the attached disk. Valid
      values are arm64 or x86_64.
    OnUpdateActionValueValuesEnum: Specifies which action to take on instance
      update with this disk. Default is to use the existing disk.

  Messages:
    LabelsValue: Labels to apply to this disk. These can be later modified by
      the disks.setLabels method. This field is only applicable for persistent
      disks.
    ResourceManagerTagsValue: Resource manager tags to be bound to the disk.
      Tag keys and values have the same definition as resource manager tags.
      Keys must be in the format `tagKeys/{tag_key_id}`, and values are in the
      format `tagValues/456`. The field is ignored (both PUT & PATCH) when
      empty.

  Fields:
    architecture: The architecture of the attached disk. Valid values are
      arm64 or x86_64.
    description: An optional description. Provide this property when creating
      the disk.
    diskName: Specifies the disk name. If not specified, the default is to use
      the name of the instance. If a disk with the same name already exists in
      the given region, the existing disk is attached to the new instance and
      the new disk is not created.
    diskSizeGb: Specifies the size of the disk in base-2 GB. The size must be
      at least 10 GB. If you specify a sourceImage, which is required for boot
      disks, the default size is the size of the sourceImage. If you do not
      specify a sourceImage, the default disk size is 500 GB.
    diskType: Specifies the disk type to use to create the instance. If not
      specified, the default is pd-standard, specified using the full URL. For
      example:
      https://www.googleapis.com/compute/v1/projects/project/zones/zone
      /diskTypes/pd-standard For a full list of acceptable values, see
      Persistent disk types. If you specify this field when creating a VM, you
      can provide either the full or partial URL. For example, the following
      values are valid: -
      https://www.googleapis.com/compute/v1/projects/project/zones/zone
      /diskTypes/diskType - projects/project/zones/zone/diskTypes/diskType -
      zones/zone/diskTypes/diskType If you specify this field when creating or
      updating an instance template or all-instances configuration, specify
      the type of the disk, not the URL. For example: pd-standard.
    enableConfidentialCompute: Whether this disk is using confidential compute
      mode.
    guestOsFeatures: A list of features to enable on the guest operating
      system. Applicable only for bootable images. Read Enabling guest
      operating system features to see a list of available options. Guest OS
      features are applied by merging initializeParams.guestOsFeatures and
      disks.guestOsFeatures
    labels: Labels to apply to this disk. These can be later modified by the
      disks.setLabels method. This field is only applicable for persistent
      disks.
    licenses: A list of publicly visible licenses. Reserved for Google's use.
    multiWriter: Indicates whether or not the disk can be read/write attached
      to more than one instance.
    onUpdateAction: Specifies which action to take on instance update with
      this disk. Default is to use the existing disk.
    provisionedIops: Indicates how many IOPS to provision for the disk. This
      sets the number of I/O operations per second that the disk can handle.
      Values must be between 10,000 and 120,000. For more details, see the
      Extreme persistent disk documentation.
    provisionedThroughput: Indicates how much throughput to provision for the
      disk. This sets the number of throughput mb per second that the disk can
      handle. Values must greater than or equal to 1.
    replicaZones: Required for each regional disk associated with the
      instance. Specify the URLs of the zones where the disk should be
      replicated to. You must provide exactly two replica zones, and one zone
      must be the same as the instance zone.
    resourceManagerTags: Resource manager tags to be bound to the disk. Tag
      keys and values have the same definition as resource manager tags. Keys
      must be in the format `tagKeys/{tag_key_id}`, and values are in the
      format `tagValues/456`. The field is ignored (both PUT & PATCH) when
      empty.
    resourcePolicies: Resource policies applied to this disk for automatic
      snapshot creations. Specified using the full or partial URL. For
      instance template, specify only the resource policy name.
    sourceImage: The source image to create this disk. When creating a new
      instance, one of initializeParams.sourceImage or
      initializeParams.sourceSnapshot or disks.source is required except for
      local SSD. To create a disk with one of the public operating system
      images, specify the image by its family name. For example, specify
      family/debian-9 to use the latest Debian 9 image: projects/debian-
      cloud/global/images/family/debian-9 Alternatively, use a specific
      version of a public operating system image: projects/debian-
      cloud/global/images/debian-9-stretch-vYYYYMMDD To create a disk with a
      custom image that you created, specify the image name in the following
      format: global/images/my-custom-image You can also specify a custom
      image by its image family, which returns the latest version of the image
      in that family. Replace the image name with family/family-name:
      global/images/family/my-image-family If the source image is deleted
      later, this field will not be set.
    sourceImageEncryptionKey: The customer-supplied encryption key of the
      source image. Required if the source image is protected by a customer-
      supplied encryption key. InstanceTemplate and InstancePropertiesPatch do
      not store customer-supplied encryption keys, so you cannot create disks
      for instances in a managed instance group if the source images are
      encrypted with your own keys.
    sourceInstantSnapshot: The source instant-snapshot to create this disk.
      When creating a new instance, one of initializeParams.sourceSnapshot or
      initializeParams.sourceInstantSnapshot initializeParams.sourceImage or
      disks.source is required except for local SSD. To create a disk with a
      snapshot that you created, specify the snapshot name in the following
      format: us-central1-a/instantSnapshots/my-backup If the source instant-
      snapshot is deleted later, this field will not be set.
    sourceSnapshot: The source snapshot to create this disk. When creating a
      new instance, one of initializeParams.sourceSnapshot or
      initializeParams.sourceImage or disks.source is required except for
      local SSD. To create a disk with a snapshot that you created, specify
      the snapshot name in the following format: global/snapshots/my-backup If
      the source snapshot is deleted later, this field will not be set.
    sourceSnapshotEncryptionKey: The customer-supplied encryption key of the
      source snapshot.
    storagePool: The storage pool in which the new disk is created. You can
      provide this as a partial or full URL to the resource. For example, the
      following are valid values: -
      https://www.googleapis.com/compute/v1/projects/project/zones/zone
      /storagePools/storagePool -
      projects/project/zones/zone/storagePools/storagePool -
      zones/zone/storagePools/storagePool
  """

    class ArchitectureValueValuesEnum(_messages.Enum):
        """The architecture of the attached disk. Valid values are arm64 or
    x86_64.

    Values:
      ARCHITECTURE_UNSPECIFIED: Default value indicating Architecture is not
        set.
      ARM64: Machines with architecture ARM64
      X86_64: Machines with architecture X86_64
    """
        ARCHITECTURE_UNSPECIFIED = 0
        ARM64 = 1
        X86_64 = 2

    class OnUpdateActionValueValuesEnum(_messages.Enum):
        """Specifies which action to take on instance update with this disk.
    Default is to use the existing disk.

    Values:
      RECREATE_DISK: Always recreate the disk.
      RECREATE_DISK_IF_SOURCE_CHANGED: Recreate the disk if source (image,
        snapshot) of this disk is different from source of existing disk.
      USE_EXISTING_DISK: Use the existing disk, this is the default behaviour.
    """
        RECREATE_DISK = 0
        RECREATE_DISK_IF_SOURCE_CHANGED = 1
        USE_EXISTING_DISK = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels to apply to this disk. These can be later modified by the
    disks.setLabels method. This field is only applicable for persistent
    disks.

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
    class ResourceManagerTagsValue(_messages.Message):
        """Resource manager tags to be bound to the disk. Tag keys and values
    have the same definition as resource manager tags. Keys must be in the
    format `tagKeys/{tag_key_id}`, and values are in the format
    `tagValues/456`. The field is ignored (both PUT & PATCH) when empty.

    Messages:
      AdditionalProperty: An additional property for a
        ResourceManagerTagsValue object.

    Fields:
      additionalProperties: Additional properties of type
        ResourceManagerTagsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ResourceManagerTagsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    architecture = _messages.EnumField('ArchitectureValueValuesEnum', 1)
    description = _messages.StringField(2)
    diskName = _messages.StringField(3)
    diskSizeGb = _messages.IntegerField(4)
    diskType = _messages.StringField(5)
    enableConfidentialCompute = _messages.BooleanField(6)
    guestOsFeatures = _messages.MessageField('GuestOsFeature', 7, repeated=True)
    labels = _messages.MessageField('LabelsValue', 8)
    licenses = _messages.StringField(9, repeated=True)
    multiWriter = _messages.BooleanField(10)
    onUpdateAction = _messages.EnumField('OnUpdateActionValueValuesEnum', 11)
    provisionedIops = _messages.IntegerField(12)
    provisionedThroughput = _messages.IntegerField(13)
    replicaZones = _messages.StringField(14, repeated=True)
    resourceManagerTags = _messages.MessageField('ResourceManagerTagsValue', 15)
    resourcePolicies = _messages.StringField(16, repeated=True)
    sourceImage = _messages.StringField(17)
    sourceImageEncryptionKey = _messages.MessageField('CustomerEncryptionKey', 18)
    sourceInstantSnapshot = _messages.StringField(19)
    sourceSnapshot = _messages.StringField(20)
    sourceSnapshotEncryptionKey = _messages.MessageField('CustomerEncryptionKey', 21)
    storagePool = _messages.StringField(22)