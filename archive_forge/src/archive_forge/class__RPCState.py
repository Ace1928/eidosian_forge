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
class _RPCState(object):
    context: contextvars.Context
    condition: threading.Condition
    due = Set[str]
    request: Any
    client: str
    initial_metadata_allowed: bool
    compression_algorithm: Optional[grpc.Compression]
    disable_next_compression: bool
    trailing_metadata: Optional[MetadataType]
    code: Optional[grpc.StatusCode]
    details: Optional[bytes]
    statused: bool
    rpc_errors: List[Exception]
    callbacks: Optional[List[NullaryCallbackType]]
    aborted: bool

    def __init__(self):
        self.context = contextvars.Context()
        self.condition = threading.Condition()
        self.due = set()
        self.request = None
        self.client = _OPEN
        self.initial_metadata_allowed = True
        self.compression_algorithm = None
        self.disable_next_compression = False
        self.trailing_metadata = None
        self.code = None
        self.details = None
        self.statused = False
        self.rpc_errors = []
        self.callbacks = []
        self.aborted = False