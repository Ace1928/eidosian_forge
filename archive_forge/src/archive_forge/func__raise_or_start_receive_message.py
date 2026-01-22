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
def _raise_or_start_receive_message(self) -> None:
    if self._state.client is _CANCELLED:
        _raise_rpc_error(self._state)
    elif not _is_rpc_state_active(self._state):
        raise StopIteration()
    else:
        self._call.start_server_batch((cygrpc.ReceiveMessageOperation(_EMPTY_FLAGS),), _receive_message(self._state, self._call, self._request_deserializer))
        self._state.due.add(_RECEIVE_MESSAGE_TOKEN)