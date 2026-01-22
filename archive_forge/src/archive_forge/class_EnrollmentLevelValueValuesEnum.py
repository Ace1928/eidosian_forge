from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class EnrollmentLevelValueValuesEnum(_messages.Enum):
    """The enrollment level of the service.

    Values:
      ENROLLMENT_LEVEL_UNSPECIFIED: Default value for proto, shouldn't be
        used.
      BLOCK_ALL: Service is enrolled in Access Approval for all requests
    """
    ENROLLMENT_LEVEL_UNSPECIFIED = 0
    BLOCK_ALL = 1