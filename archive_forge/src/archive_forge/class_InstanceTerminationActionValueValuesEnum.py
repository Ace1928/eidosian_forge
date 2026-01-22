from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceTerminationActionValueValuesEnum(_messages.Enum):
    """Optional. Specifies the termination action for the instance.

    Values:
      INSTANCE_TERMINATION_ACTION_UNSPECIFIED: Default value. This value is
        unused.
      DELETE: Delete the VM.
      STOP: Stop the VM without storing in-memory content. default action.
    """
    INSTANCE_TERMINATION_ACTION_UNSPECIFIED = 0
    DELETE = 1
    STOP = 2