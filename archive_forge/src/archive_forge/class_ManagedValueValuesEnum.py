from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedValueValuesEnum(_messages.Enum):
    """The management state of the resource as specified by the API client.

    Values:
      MANAGED_STATE_UNSPECIFIED: The management state of the resource is
        unknown or unspecified.
      MANAGED: The resource is managed.
      UNMANAGED: The resource is not managed.
    """
    MANAGED_STATE_UNSPECIFIED = 0
    MANAGED = 1
    UNMANAGED = 2