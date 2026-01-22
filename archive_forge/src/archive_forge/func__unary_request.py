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
def _unary_request(rpc_event: cygrpc.BaseEvent, state: _RPCState, request_deserializer: Optional[DeserializingFunction]) -> Callable[[], Any]:

    def unary_request():
        with state.condition:
            if not _is_rpc_state_active(state):
                return None
            else:
                rpc_event.call.start_server_batch((cygrpc.ReceiveMessageOperation(_EMPTY_FLAGS),), _receive_message(state, rpc_event.call, request_deserializer))
                state.due.add(_RECEIVE_MESSAGE_TOKEN)
                while True:
                    state.condition.wait()
                    if state.request is None:
                        if state.client is _CLOSED:
                            details = '"{}" requires exactly one request message.'.format(rpc_event.call_details.method)
                            _abort(state, rpc_event.call, cygrpc.StatusCode.unimplemented, _common.encode(details))
                            return None
                        elif state.client is _CANCELLED:
                            return None
                    else:
                        request = state.request
                        state.request = None
                        return request
    return unary_request