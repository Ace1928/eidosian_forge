from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PreservedStatePreservedDisk(_messages.Message):
    """A PreservedStatePreservedDisk object.

  Enums:
    AutoDeleteValueValuesEnum: These stateful disks will never be deleted
      during autohealing, update, instance recreate operations. This flag is
      used to configure if the disk should be deleted after it is no longer
      used by the group, e.g. when the given instance or the whole MIG is
      deleted. Note: disks attached in READ_ONLY mode cannot be auto-deleted.
    ModeValueValuesEnum: The mode in which to attach this disk, either
      READ_WRITE or READ_ONLY. If not specified, the default is to attach the
      disk in READ_WRITE mode.

  Fields:
    autoDelete: These stateful disks will never be deleted during autohealing,
      update, instance recreate operations. This flag is used to configure if
      the disk should be deleted after it is no longer used by the group, e.g.
      when the given instance or the whole MIG is deleted. Note: disks
      attached in READ_ONLY mode cannot be auto-deleted.
    mode: The mode in which to attach this disk, either READ_WRITE or
      READ_ONLY. If not specified, the default is to attach the disk in
      READ_WRITE mode.
    source: The URL of the disk resource that is stateful and should be
      attached to the VM instance.
  """

    class AutoDeleteValueValuesEnum(_messages.Enum):
        """These stateful disks will never be deleted during autohealing, update,
    instance recreate operations. This flag is used to configure if the disk
    should be deleted after it is no longer used by the group, e.g. when the
    given instance or the whole MIG is deleted. Note: disks attached in
    READ_ONLY mode cannot be auto-deleted.

    Values:
      NEVER: <no description>
      ON_PERMANENT_INSTANCE_DELETION: <no description>
    """
        NEVER = 0
        ON_PERMANENT_INSTANCE_DELETION = 1

    class ModeValueValuesEnum(_messages.Enum):
        """The mode in which to attach this disk, either READ_WRITE or READ_ONLY.
    If not specified, the default is to attach the disk in READ_WRITE mode.

    Values:
      READ_ONLY: Attaches this disk in read-only mode. Multiple VM instances
        can use a disk in READ_ONLY mode at a time.
      READ_WRITE: *[Default]* Attaches this disk in READ_WRITE mode. Only one
        VM instance at a time can be attached to a disk in READ_WRITE mode.
    """
        READ_ONLY = 0
        READ_WRITE = 1
    autoDelete = _messages.EnumField('AutoDeleteValueValuesEnum', 1)
    mode = _messages.EnumField('ModeValueValuesEnum', 2)
    source = _messages.StringField(3)