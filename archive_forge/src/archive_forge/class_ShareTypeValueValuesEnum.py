from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ShareTypeValueValuesEnum(_messages.Enum):
    """Type of sharing for this shared-reservation

    Values:
      LOCAL: Default value.
      ORGANIZATION: Shared-reservation is open to entire Organization
      SHARE_TYPE_UNSPECIFIED: Default value. This value is unused.
      SPECIFIC_PROJECTS: Shared-reservation is open to specific projects
    """
    LOCAL = 0
    ORGANIZATION = 1
    SHARE_TYPE_UNSPECIFIED = 2
    SPECIFIC_PROJECTS = 3