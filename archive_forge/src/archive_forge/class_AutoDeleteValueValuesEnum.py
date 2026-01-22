from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutoDeleteValueValuesEnum(_messages.Enum):
    """These stateful IPs will never be released during autohealing, update
    or VM instance recreate operations. This flag is used to configure if the
    IP reservation should be deleted after it is no longer used by the group,
    e.g. when the given instance or the whole group is deleted.

    Values:
      NEVER: <no description>
      ON_PERMANENT_INSTANCE_DELETION: <no description>
    """
    NEVER = 0
    ON_PERMANENT_INSTANCE_DELETION = 1