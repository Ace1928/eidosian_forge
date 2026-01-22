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
def _receive_close_on_server(state: _RPCState) -> ServerCallbackTag:

    def receive_close_on_server(receive_close_on_server_event):
        with state.condition:
            if receive_close_on_server_event.batch_operations[0].cancelled():
                state.client = _CANCELLED
            elif state.client is _OPEN:
                state.client = _CLOSED
            state.condition.notify_all()
            return _possibly_finish_call(state, _RECEIVE_CLOSE_ON_SERVER_TOKEN)
    return receive_close_on_server