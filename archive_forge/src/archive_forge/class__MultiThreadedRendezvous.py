import copy
import functools
import logging
import os
import sys
import threading
import time
import types
from typing import (
import grpc  # pytype: disable=pyi-error
from grpc import _common  # pytype: disable=pyi-error
from grpc import _compression  # pytype: disable=pyi-error
from grpc import _grpcio_metadata  # pytype: disable=pyi-error
from grpc import _observability  # pytype: disable=pyi-error
from grpc._cython import cygrpc
from grpc._typing import ChannelArgumentType
from grpc._typing import DeserializingFunction
from grpc._typing import IntegratedCallFactory
from grpc._typing import MetadataType
from grpc._typing import NullaryCallbackType
from grpc._typing import ResponseType
from grpc._typing import SerializingFunction
from grpc._typing import UserTag
import grpc.experimental  # pytype: disable=pyi-error
class _MultiThreadedRendezvous(_Rendezvous, grpc.Call, grpc.Future):
    """An RPC iterator that depends on a channel spin thread.

    This iterator relies upon a per-channel thread running in the background,
    dequeueing events from the completion queue, and notifying threads waiting
    on the threading.Condition object in the _RPCState object.

    This extra thread allows _MultiThreadedRendezvous to fulfill the grpc.Future interface
    and to mediate a bidirection streaming RPC.
    """
    _state: _RPCState

    def initial_metadata(self) -> Optional[MetadataType]:
        """See grpc.Call.initial_metadata"""
        with self._state.condition:

            def _done():
                return self._state.initial_metadata is not None
            _common.wait(self._state.condition.wait, _done)
            return self._state.initial_metadata

    def trailing_metadata(self) -> Optional[MetadataType]:
        """See grpc.Call.trailing_metadata"""
        with self._state.condition:

            def _done():
                return self._state.trailing_metadata is not None
            _common.wait(self._state.condition.wait, _done)
            return self._state.trailing_metadata

    def code(self) -> Optional[grpc.StatusCode]:
        """See grpc.Call.code"""
        with self._state.condition:

            def _done():
                return self._state.code is not None
            _common.wait(self._state.condition.wait, _done)
            return self._state.code

    def details(self) -> Optional[str]:
        """See grpc.Call.details"""
        with self._state.condition:

            def _done():
                return self._state.details is not None
            _common.wait(self._state.condition.wait, _done)
            return _common.decode(self._state.details)

    def debug_error_string(self) -> Optional[str]:
        with self._state.condition:

            def _done():
                return self._state.debug_error_string is not None
            _common.wait(self._state.condition.wait, _done)
            return _common.decode(self._state.debug_error_string)

    def cancelled(self) -> bool:
        with self._state.condition:
            return self._state.cancelled

    def running(self) -> bool:
        with self._state.condition:
            return self._state.code is None

    def done(self) -> bool:
        with self._state.condition:
            return self._state.code is not None

    def _is_complete(self) -> bool:
        return self._state.code is not None

    def result(self, timeout: Optional[float]=None) -> Any:
        """Returns the result of the computation or raises its exception.

        See grpc.Future.result for the full API contract.
        """
        with self._state.condition:
            timed_out = _common.wait(self._state.condition.wait, self._is_complete, timeout=timeout)
            if timed_out:
                raise grpc.FutureTimeoutError()
            elif self._state.code is grpc.StatusCode.OK:
                return self._state.response
            elif self._state.cancelled:
                raise grpc.FutureCancelledError()
            else:
                raise self

    def exception(self, timeout: Optional[float]=None) -> Optional[Exception]:
        """Return the exception raised by the computation.

        See grpc.Future.exception for the full API contract.
        """
        with self._state.condition:
            timed_out = _common.wait(self._state.condition.wait, self._is_complete, timeout=timeout)
            if timed_out:
                raise grpc.FutureTimeoutError()
            elif self._state.code is grpc.StatusCode.OK:
                return None
            elif self._state.cancelled:
                raise grpc.FutureCancelledError()
            else:
                return self

    def traceback(self, timeout: Optional[float]=None) -> Optional[types.TracebackType]:
        """Access the traceback of the exception raised by the computation.

        See grpc.future.traceback for the full API contract.
        """
        with self._state.condition:
            timed_out = _common.wait(self._state.condition.wait, self._is_complete, timeout=timeout)
            if timed_out:
                raise grpc.FutureTimeoutError()
            elif self._state.code is grpc.StatusCode.OK:
                return None
            elif self._state.cancelled:
                raise grpc.FutureCancelledError()
            else:
                try:
                    raise self
                except grpc.RpcError:
                    return sys.exc_info()[2]

    def add_done_callback(self, fn: Callable[[grpc.Future], None]) -> None:
        with self._state.condition:
            if self._state.code is None:
                self._state.callbacks.append(functools.partial(fn, self))
                return
        fn(self)

    def _next(self) -> Any:
        with self._state.condition:
            if self._state.code is None:
                event_handler = _event_handler(self._state, self._response_deserializer)
                self._state.due.add(cygrpc.OperationType.receive_message)
                operating = self._call.operate((cygrpc.ReceiveMessageOperation(_EMPTY_FLAGS),), event_handler)
                if not operating:
                    self._state.due.remove(cygrpc.OperationType.receive_message)
            elif self._state.code is grpc.StatusCode.OK:
                raise StopIteration()
            else:
                raise self

            def _response_ready():
                return self._state.response is not None or (cygrpc.OperationType.receive_message not in self._state.due and self._state.code is not None)
            _common.wait(self._state.condition.wait, _response_ready)
            if self._state.response is not None:
                response = self._state.response
                self._state.response = None
                return response
            elif cygrpc.OperationType.receive_message not in self._state.due:
                if self._state.code is grpc.StatusCode.OK:
                    raise StopIteration()
                elif self._state.code is not None:
                    raise self