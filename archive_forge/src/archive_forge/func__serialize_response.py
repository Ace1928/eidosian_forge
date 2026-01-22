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
def _serialize_response(rpc_event: cygrpc.BaseEvent, state: _RPCState, response: Any, response_serializer: Optional[SerializingFunction]) -> Optional[bytes]:
    serialized_response = _common.serialize(response, response_serializer)
    if serialized_response is None:
        with state.condition:
            _abort(state, rpc_event.call, cygrpc.StatusCode.internal, b'Failed to serialize response!')
        return None
    else:
        return serialized_response