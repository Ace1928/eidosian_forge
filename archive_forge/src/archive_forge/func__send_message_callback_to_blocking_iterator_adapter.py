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
def _send_message_callback_to_blocking_iterator_adapter(rpc_event: cygrpc.BaseEvent, state: _RPCState, send_response_callback: Callable[[ResponseType], None], response_iterator: Iterator[ResponseType]) -> None:
    while True:
        response, proceed = _take_response_from_response_iterator(rpc_event, state, response_iterator)
        if proceed:
            send_response_callback(response)
            if not _is_rpc_state_active(state):
                break
        else:
            break