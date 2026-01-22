from __future__ import annotations
import collections
from concurrent import futures
import contextvars
import enum
import logging
import threading
import time
import traceback
from typing import (
import grpc  # pytype: disable=pyi-error
from grpc import _common  # pytype: disable=pyi-error
from grpc import _compression  # pytype: disable=pyi-error
from grpc import _interceptor  # pytype: disable=pyi-error
from grpc._cython import cygrpc
from grpc._typing import ArityAgnosticMethodHandler
from grpc._typing import ChannelArgumentType
from grpc._typing import DeserializingFunction
from grpc._typing import MetadataType
from grpc._typing import NullaryCallbackType
from grpc._typing import ResponseType
from grpc._typing import SerializingFunction
from grpc._typing import ServerCallbackTag
from grpc._typing import ServerTagCallbackType
def _reject_rpc(rpc_event: cygrpc.BaseEvent, rpc_state: _RPCState, status: cygrpc.StatusCode, details: bytes):
    operations = (_get_initial_metadata_operation(rpc_state, None), cygrpc.ReceiveCloseOnServerOperation(_EMPTY_FLAGS), cygrpc.SendStatusFromServerOperation(None, status, details, _EMPTY_FLAGS))
    rpc_event.call.start_server_batch(operations, lambda ignored_event: (rpc_state, ()))