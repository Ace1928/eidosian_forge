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
def _handle_unary_unary(rpc_event: cygrpc.BaseEvent, state: _RPCState, method_handler: grpc.RpcMethodHandler, default_thread_pool: futures.ThreadPoolExecutor) -> futures.Future:
    unary_request = _unary_request(rpc_event, state, method_handler.request_deserializer)
    thread_pool = _select_thread_pool_for_behavior(method_handler.unary_unary, default_thread_pool)
    return thread_pool.submit(state.context.run, _unary_response_in_pool, rpc_event, state, method_handler.unary_unary, unary_request, method_handler.request_deserializer, method_handler.response_serializer)