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
def _receive_message(state: _RPCState, call: cygrpc.Call, request_deserializer: Optional[DeserializingFunction]) -> ServerCallbackTag:

    def receive_message(receive_message_event):
        serialized_request = _serialized_request(receive_message_event)
        if serialized_request is None:
            with state.condition:
                if state.client is _OPEN:
                    state.client = _CLOSED
                state.condition.notify_all()
                return _possibly_finish_call(state, _RECEIVE_MESSAGE_TOKEN)
        else:
            request = _common.deserialize(serialized_request, request_deserializer)
            with state.condition:
                if request is None:
                    _abort(state, call, cygrpc.StatusCode.internal, b'Exception deserializing request!')
                else:
                    state.request = request
                state.condition.notify_all()
                return _possibly_finish_call(state, _RECEIVE_MESSAGE_TOKEN)
    return receive_message