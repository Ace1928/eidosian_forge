from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedSidecarsValueListEntryValuesEnum(_messages.Enum):
    """ManagedSidecarsValueListEntryValuesEnum enum type.

    Values:
      MANAGED_SIDECAR_UNSPECIFIED: Default enum type; should not be used.
      PRIVILEGED_DOCKER_DAEMON: Sidecar for a privileged docker daemon.
    """
    MANAGED_SIDECAR_UNSPECIFIED = 0
    PRIVILEGED_DOCKER_DAEMON = 1