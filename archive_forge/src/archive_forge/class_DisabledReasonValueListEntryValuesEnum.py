from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DisabledReasonValueListEntryValuesEnum(_messages.Enum):
    """DisabledReasonValueListEntryValuesEnum enum type.

    Values:
      DISABLED_REASON_UNSPECIFIED: This is an unknown reason for disabling.
      KMS_KEY_ISSUE: The KMS key used by the instance is either revoked or
        denied access to
    """
    DISABLED_REASON_UNSPECIFIED = 0
    KMS_KEY_ISSUE = 1