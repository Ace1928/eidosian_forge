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
class _ServerState(object):
    lock: threading.RLock
    completion_queue: cygrpc.CompletionQueue
    server: cygrpc.Server
    generic_handlers: List[grpc.GenericRpcHandler]
    interceptor_pipeline: Optional[_interceptor._ServicePipeline]
    thread_pool: futures.ThreadPoolExecutor
    stage: _ServerStage
    termination_event: threading.Event
    shutdown_events: List[threading.Event]
    maximum_concurrent_rpcs: Optional[int]
    active_rpc_count: int
    rpc_states: Set[_RPCState]
    due: Set[str]
    server_deallocated: bool

    def __init__(self, completion_queue: cygrpc.CompletionQueue, server: cygrpc.Server, generic_handlers: Sequence[grpc.GenericRpcHandler], interceptor_pipeline: Optional[_interceptor._ServicePipeline], thread_pool: futures.ThreadPoolExecutor, maximum_concurrent_rpcs: Optional[int]):
        self.lock = threading.RLock()
        self.completion_queue = completion_queue
        self.server = server
        self.generic_handlers = list(generic_handlers)
        self.interceptor_pipeline = interceptor_pipeline
        self.thread_pool = thread_pool
        self.stage = _ServerStage.STOPPED
        self.termination_event = threading.Event()
        self.shutdown_events = [self.termination_event]
        self.maximum_concurrent_rpcs = maximum_concurrent_rpcs
        self.active_rpc_count = 0
        self.rpc_states = set()
        self.due = set()
        self.server_deallocated = False