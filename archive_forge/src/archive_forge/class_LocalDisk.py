from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LocalDisk(_messages.Message):
    """A Local attached disk resource.

  Fields:
    autoDelete: Optional. Output only. Specifies whether the disk will be
      auto-deleted when the instance is deleted (but not when the disk is
      detached from the instance).
    boot: Optional. Output only. Indicates that this is a boot disk. The
      virtual machine will use the first partition of the disk for its root
      filesystem.
    deviceName: Optional. Output only. Specifies a unique device name of your
      choice that is reflected into the `/dev/disk/by-id/google-*` tree of a
      Linux operating system running within the instance. This name can be
      used to reference the device for mounting, resizing, and so on, from
      within the instance. If not specified, the server chooses a default
      device name to apply to this disk, in the form persistent-disk-x, where
      x is a number assigned by Google Compute Engine. This field is only
      applicable for persistent disks.
    guestOsFeatures: Output only. Indicates a list of features to enable on
      the guest operating system. Applicable only for bootable images. Read
      Enabling guest operating system features to see a list of available
      options.
    index: Output only. A zero-based index to this disk, where 0 is reserved
      for the boot disk. If you have many disks attached to an instance, each
      disk would have a unique index number.
    initializeParams: Input only. Specifies the parameters for a new disk that
      will be created alongside the new instance. Use initialization
      parameters to create boot disks or local SSDs attached to the new
      instance. This property is mutually exclusive with the source property;
      you can only define one or the other, but not both.
    interface: Specifies the disk interface to use for attaching this disk,
      which is either SCSI or NVME. The default is SCSI. Persistent disks must
      always use SCSI and the request will fail if you attempt to attach a
      persistent disk in any other format than SCSI. Local SSDs can use either
      NVME or SCSI. For performance characteristics of SCSI over NVMe, see
      Local SSD performance. Valid values: * `NVME` * `SCSI`
    kind: Output only. Type of the resource. Always compute#attachedDisk for
      attached disks.
    licenses: Output only. Any valid publicly visible licenses.
    mode: The mode in which to attach this disk, either `READ_WRITE` or
      `READ_ONLY`. If not specified, the default is to attach the disk in
      `READ_WRITE` mode. Valid values: * `READ_ONLY` * `READ_WRITE`
    source: Specifies a valid partial or full URL to an existing Persistent
      Disk resource.
    type: Specifies the type of the disk, either `SCRATCH` or `PERSISTENT`. If
      not specified, the default is `PERSISTENT`. Valid values: * `PERSISTENT`
      * `SCRATCH`
  """
    autoDelete = _messages.BooleanField(1)
    boot = _messages.BooleanField(2)
    deviceName = _messages.StringField(3)
    guestOsFeatures = _messages.MessageField('RuntimeGuestOsFeature', 4, repeated=True)
    index = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    initializeParams = _messages.MessageField('LocalDiskInitializeParams', 6)
    interface = _messages.StringField(7)
    kind = _messages.StringField(8)
    licenses = _messages.StringField(9, repeated=True)
    mode = _messages.StringField(10)
    source = _messages.StringField(11)
    type = _messages.StringField(12)