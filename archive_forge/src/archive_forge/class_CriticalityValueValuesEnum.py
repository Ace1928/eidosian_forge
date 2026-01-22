from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CriticalityValueValuesEnum(_messages.Enum):
    """Optional. Criticality of this workload.

    Values:
      CRITICALITY_UNSPECIFIED: Default. Resource is not supported and is not
        expected to provide any guarantees.
      MISSION_CRITICAL: The resource is mission-critical to the organization.
      HIGH: The resource may not directly affect the mission of a specific
        unit, but is of high importance to the organization.
      MEDIUM: The resource is of medium importance to the organization.
      LOW: The resource is of low importance to the organization.
    """
    CRITICALITY_UNSPECIFIED = 0
    MISSION_CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4