from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AbuseValueValuesEnum(_messages.Enum):
    """AbuseValueValuesEnum enum type.

    Values:
      ABUSE_UNKNOWN_REASON: An unknown reason indicates that the abuse system
        has not sent a signal for this container.
      ABUSE_CONTROL_PLANE_SYNC: Due to various reasons CCFE might proactively
        restate a container state to a CLH to ensure that the CLH and CCFE are
        both aware of the container state. This reason can be tied to any of
        the states.
      SUSPEND: If a container is deemed abusive we receive a suspend signal.
        Suspend is a reason to put the container into an INTERNAL_OFF state.
      REINSTATE: Containers that were once considered abusive can later be
        deemed non-abusive. When this happens we must reinstate the container.
        Reinstate is a reason to put the container into an ON state.
    """
    ABUSE_UNKNOWN_REASON = 0
    ABUSE_CONTROL_PLANE_SYNC = 1
    SUSPEND = 2
    REINSTATE = 3