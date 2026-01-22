from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagerTypeValueValuesEnum(_messages.Enum):
    """Stores extra information about what Google resource is directly
    responsible for a given Workload resource.

    Values:
      TYPE_UNSPECIFIED: Default. Should not be used.
      GKE_HUB: Resource managed by GKE Hub.
      BACKEND_SERVICE: Resource managed by Arcus, Backend Service
    """
    TYPE_UNSPECIFIED = 0
    GKE_HUB = 1
    BACKEND_SERVICE = 2