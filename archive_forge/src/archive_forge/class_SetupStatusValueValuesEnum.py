from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SetupStatusValueValuesEnum(_messages.Enum):
    """Indicates SAA enrollment status of a given workload.

    Values:
      SETUP_STATE_UNSPECIFIED: Unspecified.
      STATUS_PENDING: SAA enrollment pending.
      STATUS_COMPLETE: SAA enrollment comopleted.
    """
    SETUP_STATE_UNSPECIFIED = 0
    STATUS_PENDING = 1
    STATUS_COMPLETE = 2