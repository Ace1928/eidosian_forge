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
def _call_behavior(rpc_event: cygrpc.BaseEvent, state: _RPCState, behavior: ArityAgnosticMethodHandler, argument: Any, request_deserializer: Optional[DeserializingFunction], send_response_callback: Optional[Callable[[ResponseType], None]]=None) -> Tuple[Union[ResponseType, Iterator[ResponseType]], bool]:
    from grpc import _create_servicer_context
    with _create_servicer_context(rpc_event, state, request_deserializer) as context:
        try:
            response_or_iterator = None
            if send_response_callback is not None:
                response_or_iterator = behavior(argument, context, send_response_callback)
            else:
                response_or_iterator = behavior(argument, context)
            return (response_or_iterator, True)
        except Exception as exception:
            with state.condition:
                if state.aborted:
                    _abort(state, rpc_event.call, cygrpc.StatusCode.unknown, b'RPC Aborted')
                elif exception not in state.rpc_errors:
                    try:
                        details = 'Exception calling application: {}'.format(exception)
                    except Exception:
                        details = 'Calling application raised unprintable Exception!'
                        _LOGGER.exception(traceback.format_exception(type(exception), exception, exception.__traceback__))
                        traceback.print_exc()
                    _LOGGER.exception(details)
                    _abort(state, rpc_event.call, cygrpc.StatusCode.unknown, _common.encode(details))
            return (None, False)