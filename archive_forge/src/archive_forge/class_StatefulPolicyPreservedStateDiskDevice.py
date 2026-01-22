from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StatefulPolicyPreservedStateDiskDevice(_messages.Message):
    """A StatefulPolicyPreservedStateDiskDevice object.

  Enums:
    AutoDeleteValueValuesEnum: These stateful disks will never be deleted
      during autohealing, update or VM instance recreate operations. This flag
      is used to configure if the disk should be deleted after it is no longer
      used by the group, e.g. when the given instance or the whole group is
      deleted. Note: disks attached in READ_ONLY mode cannot be auto-deleted.

  Fields:
    autoDelete: These stateful disks will never be deleted during autohealing,
      update or VM instance recreate operations. This flag is used to
      configure if the disk should be deleted after it is no longer used by
      the group, e.g. when the given instance or the whole group is deleted.
      Note: disks attached in READ_ONLY mode cannot be auto-deleted.
  """

    class AutoDeleteValueValuesEnum(_messages.Enum):
        """These stateful disks will never be deleted during autohealing, update
    or VM instance recreate operations. This flag is used to configure if the
    disk should be deleted after it is no longer used by the group, e.g. when
    the given instance or the whole group is deleted. Note: disks attached in
    READ_ONLY mode cannot be auto-deleted.

    Values:
      NEVER: <no description>
      ON_PERMANENT_INSTANCE_DELETION: <no description>
    """
        NEVER = 0
        ON_PERMANENT_INSTANCE_DELETION = 1
    autoDelete = _messages.EnumField('AutoDeleteValueValuesEnum', 1)