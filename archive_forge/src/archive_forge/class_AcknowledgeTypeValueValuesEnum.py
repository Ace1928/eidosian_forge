from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AcknowledgeTypeValueValuesEnum(_messages.Enum):
    """Optional. Acknowledge type of specified violation.

    Values:
      ACKNOWLEDGE_TYPE_UNSPECIFIED: Acknowledge type unspecified.
      SINGLE_VIOLATION: Acknowledge only the specific violation.
      EXISTING_CHILD_RESOURCE_VIOLATIONS: Acknowledge specified orgPolicy
        violation and also associated resource violations.
    """
    ACKNOWLEDGE_TYPE_UNSPECIFIED = 0
    SINGLE_VIOLATION = 1
    EXISTING_CHILD_RESOURCE_VIOLATIONS = 2