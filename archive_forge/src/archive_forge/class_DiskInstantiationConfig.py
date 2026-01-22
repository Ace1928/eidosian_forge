from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DiskInstantiationConfig(_messages.Message):
    """A specification of the desired way to instantiate a disk in the instance
  template when its created from a source instance.

  Enums:
    InstantiateFromValueValuesEnum: Specifies whether to include the disk and
      what image to use. Possible values are: - source-image: to use the same
      image that was used to create the source instance's corresponding disk.
      Applicable to the boot disk and additional read-write disks. - source-
      image-family: to use the same image family that was used to create the
      source instance's corresponding disk. Applicable to the boot disk and
      additional read-write disks. - custom-image: to use a user-provided
      image url for disk creation. Applicable to the boot disk and additional
      read-write disks. - attach-read-only: to attach a read-only disk.
      Applicable to read-only disks. - do-not-include: to exclude a disk from
      the template. Applicable to additional read-write disks, local SSDs, and
      read-only disks.

  Fields:
    autoDelete: Specifies whether the disk will be auto-deleted when the
      instance is deleted (but not when the disk is detached from the
      instance).
    customImage: The custom source image to be used to restore this disk when
      instantiating this instance template.
    deviceName: Specifies the device name of the disk to which the
      configurations apply to.
    instantiateFrom: Specifies whether to include the disk and what image to
      use. Possible values are: - source-image: to use the same image that was
      used to create the source instance's corresponding disk. Applicable to
      the boot disk and additional read-write disks. - source-image-family: to
      use the same image family that was used to create the source instance's
      corresponding disk. Applicable to the boot disk and additional read-
      write disks. - custom-image: to use a user-provided image url for disk
      creation. Applicable to the boot disk and additional read-write disks. -
      attach-read-only: to attach a read-only disk. Applicable to read-only
      disks. - do-not-include: to exclude a disk from the template. Applicable
      to additional read-write disks, local SSDs, and read-only disks.
  """

    class InstantiateFromValueValuesEnum(_messages.Enum):
        """Specifies whether to include the disk and what image to use. Possible
    values are: - source-image: to use the same image that was used to create
    the source instance's corresponding disk. Applicable to the boot disk and
    additional read-write disks. - source-image-family: to use the same image
    family that was used to create the source instance's corresponding disk.
    Applicable to the boot disk and additional read-write disks. - custom-
    image: to use a user-provided image url for disk creation. Applicable to
    the boot disk and additional read-write disks. - attach-read-only: to
    attach a read-only disk. Applicable to read-only disks. - do-not-include:
    to exclude a disk from the template. Applicable to additional read-write
    disks, local SSDs, and read-only disks.

    Values:
      ATTACH_READ_ONLY: Attach the existing disk in read-only mode. The
        request will fail if the disk was attached in read-write mode on the
        source instance. Applicable to: read-only disks.
      BLANK: Create a blank disk. The disk will be created unformatted.
        Applicable to: additional read-write disks, local SSDs.
      CUSTOM_IMAGE: Use the custom image specified in the custom_image field.
        Applicable to: boot disk, additional read-write disks.
      DEFAULT: Use the default instantiation option for the corresponding type
        of disk. For boot disk and any other R/W disks, new custom images will
        be created from each disk. For read-only disks, they will be attached
        in read-only mode. Local SSD disks will be created as blank volumes.
      DO_NOT_INCLUDE: Do not include the disk in the instance template.
        Applicable to: additional read-write disks, local SSDs, read-only
        disks.
      SOURCE_IMAGE: Use the same source image used for creation of the source
        instance's corresponding disk. The request will fail if the source
        VM's disk was created from a snapshot. Applicable to: boot disk,
        additional read-write disks.
      SOURCE_IMAGE_FAMILY: Use the same source image family used for creation
        of the source instance's corresponding disk. The request will fail if
        the source image of the source disk does not belong to any image
        family. Applicable to: boot disk, additional read-write disks.
    """
        ATTACH_READ_ONLY = 0
        BLANK = 1
        CUSTOM_IMAGE = 2
        DEFAULT = 3
        DO_NOT_INCLUDE = 4
        SOURCE_IMAGE = 5
        SOURCE_IMAGE_FAMILY = 6
    autoDelete = _messages.BooleanField(1)
    customImage = _messages.StringField(2)
    deviceName = _messages.StringField(3)
    instantiateFrom = _messages.EnumField('InstantiateFromValueValuesEnum', 4)