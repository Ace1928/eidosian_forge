from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KajEnrollmentStateValueValuesEnum(_messages.Enum):
    """Output only. Represents the KAJ enrollment state of the given
    workload.

    Values:
      KAJ_ENROLLMENT_STATE_UNSPECIFIED: Default State for KAJ Enrollment.
      KAJ_ENROLLMENT_STATE_PENDING: Pending State for KAJ Enrollment.
      KAJ_ENROLLMENT_STATE_COMPLETE: Complete State for KAJ Enrollment.
    """
    KAJ_ENROLLMENT_STATE_UNSPECIFIED = 0
    KAJ_ENROLLMENT_STATE_PENDING = 1
    KAJ_ENROLLMENT_STATE_COMPLETE = 2