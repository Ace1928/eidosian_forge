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
def _process_event_and_continue(state: _ServerState, event: cygrpc.BaseEvent) -> bool:
    should_continue = True
    if event.tag is _SHUTDOWN_TAG:
        with state.lock:
            state.due.remove(_SHUTDOWN_TAG)
            if _stop_serving(state):
                should_continue = False
    elif event.tag is _REQUEST_CALL_TAG:
        with state.lock:
            state.due.remove(_REQUEST_CALL_TAG)
            concurrency_exceeded = state.maximum_concurrent_rpcs is not None and state.active_rpc_count >= state.maximum_concurrent_rpcs
            rpc_state, rpc_future = _handle_call(event, state.generic_handlers, state.interceptor_pipeline, state.thread_pool, concurrency_exceeded)
            if rpc_state is not None:
                state.rpc_states.add(rpc_state)
            if rpc_future is not None:
                state.active_rpc_count += 1
                rpc_future.add_done_callback(lambda unused_future: _on_call_completed(state))
            if state.stage is _ServerStage.STARTED:
                _request_call(state)
            elif _stop_serving(state):
                should_continue = False
    else:
        rpc_state, callbacks = event.tag(event)
        for callback in callbacks:
            try:
                callback()
            except Exception:
                _LOGGER.exception('Exception calling callback!')
        if rpc_state is not None:
            with state.lock:
                state.rpc_states.remove(rpc_state)
                if _stop_serving(state):
                    should_continue = False
    return should_continue