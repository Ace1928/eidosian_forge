import copy
import functools
import logging
import os
import sys
import threading
import time
import types
from typing import (
import grpc  # pytype: disable=pyi-error
from grpc import _common  # pytype: disable=pyi-error
from grpc import _compression  # pytype: disable=pyi-error
from grpc import _grpcio_metadata  # pytype: disable=pyi-error
from grpc import _observability  # pytype: disable=pyi-error
from grpc._cython import cygrpc
from grpc._typing import ChannelArgumentType
from grpc._typing import DeserializingFunction
from grpc._typing import IntegratedCallFactory
from grpc._typing import MetadataType
from grpc._typing import NullaryCallbackType
from grpc._typing import ResponseType
from grpc._typing import SerializingFunction
from grpc._typing import UserTag
import grpc.experimental  # pytype: disable=pyi-error
def _spawn_delivery(state: _ChannelConnectivityState, callbacks: Sequence[Callable[[grpc.ChannelConnectivity], None]]) -> None:
    delivering_thread = cygrpc.ForkManagedThread(target=_deliver, args=(state, state.connectivity, callbacks))
    delivering_thread.setDaemon(True)
    delivering_thread.start()
    state.delivering = True