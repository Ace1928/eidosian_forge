from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TemporaryHoldValueValuesEnum(_messages.Enum):
    """Specifies how each object's temporary hold status should be preserved
    for transfers between Google Cloud Storage buckets. If unspecified, the
    default behavior is the same as TEMPORARY_HOLD_PRESERVE.

    Values:
      TEMPORARY_HOLD_UNSPECIFIED: Temporary hold behavior is unspecified.
      TEMPORARY_HOLD_SKIP: Do not set a temporary hold on the destination
        object.
      TEMPORARY_HOLD_PRESERVE: Preserve the object's original temporary hold
        status.
    """
    TEMPORARY_HOLD_UNSPECIFIED = 0
    TEMPORARY_HOLD_SKIP = 1
    TEMPORARY_HOLD_PRESERVE = 2