import abc
import contextlib
import enum
import logging
import sys
from grpc import _compression
from grpc._cython import cygrpc as _cygrpc
from grpc._runtime_protos import protos
from grpc._runtime_protos import protos_and_services
from grpc._runtime_protos import services
@enum.unique
class ChannelConnectivity(enum.Enum):
    """Mirrors grpc_connectivity_state in the gRPC Core.

    Attributes:
      IDLE: The channel is idle.
      CONNECTING: The channel is connecting.
      READY: The channel is ready to conduct RPCs.
      TRANSIENT_FAILURE: The channel has seen a failure from which it expects
        to recover.
      SHUTDOWN: The channel has seen a failure from which it cannot recover.
    """
    IDLE = (_cygrpc.ConnectivityState.idle, 'idle')
    CONNECTING = (_cygrpc.ConnectivityState.connecting, 'connecting')
    READY = (_cygrpc.ConnectivityState.ready, 'ready')
    TRANSIENT_FAILURE = (_cygrpc.ConnectivityState.transient_failure, 'transient failure')
    SHUTDOWN = (_cygrpc.ConnectivityState.shutdown, 'shutdown')