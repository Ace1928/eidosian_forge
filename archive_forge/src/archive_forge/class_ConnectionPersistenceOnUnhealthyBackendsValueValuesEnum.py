from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectionPersistenceOnUnhealthyBackendsValueValuesEnum(_messages.Enum):
    """Specifies connection persistence when backends are unhealthy. The
    default value is DEFAULT_FOR_PROTOCOL. If set to DEFAULT_FOR_PROTOCOL, the
    existing connections persist on unhealthy backends only for connection-
    oriented protocols (TCP and SCTP) and only if the Tracking Mode is
    PER_CONNECTION (default tracking mode) or the Session Affinity is
    configured for 5-tuple. They do not persist for UDP. If set to
    NEVER_PERSIST, after a backend becomes unhealthy, the existing connections
    on the unhealthy backend are never persisted on the unhealthy backend.
    They are always diverted to newly selected healthy backends (unless all
    backends are unhealthy). If set to ALWAYS_PERSIST, existing connections
    always persist on unhealthy backends regardless of protocol and session
    affinity. It is generally not recommended to use this mode overriding the
    default. For more details, see [Connection Persistence for Network Load
    Balancing](https://cloud.google.com/load-balancing/docs/network/networklb-
    backend-service#connection-persistence) and [Connection Persistence for
    Internal TCP/UDP Load Balancing](https://cloud.google.com/load-
    balancing/docs/internal#connection-persistence).

    Values:
      ALWAYS_PERSIST: <no description>
      DEFAULT_FOR_PROTOCOL: <no description>
      NEVER_PERSIST: <no description>
    """
    ALWAYS_PERSIST = 0
    DEFAULT_FOR_PROTOCOL = 1
    NEVER_PERSIST = 2