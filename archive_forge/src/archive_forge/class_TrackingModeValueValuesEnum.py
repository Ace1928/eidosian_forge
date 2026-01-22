from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TrackingModeValueValuesEnum(_messages.Enum):
    """Specifies the key used for connection tracking. There are two options:
    - PER_CONNECTION: This is the default mode. The Connection Tracking is
    performed as per the Connection Key (default Hash Method) for the specific
    protocol. - PER_SESSION: The Connection Tracking is performed as per the
    configured Session Affinity. It matches the configured Session Affinity.
    For more details, see [Tracking Mode for Network Load
    Balancing](https://cloud.google.com/load-balancing/docs/network/networklb-
    backend-service#tracking-mode) and [Tracking Mode for Internal TCP/UDP
    Load Balancing](https://cloud.google.com/load-
    balancing/docs/internal#tracking-mode).

    Values:
      INVALID_TRACKING_MODE: <no description>
      PER_CONNECTION: <no description>
      PER_SESSION: <no description>
    """
    INVALID_TRACKING_MODE = 0
    PER_CONNECTION = 1
    PER_SESSION = 2