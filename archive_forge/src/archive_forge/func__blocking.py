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
def _blocking(self, request_iterator: Iterator, timeout: Optional[float], metadata: Optional[MetadataType], credentials: Optional[grpc.CallCredentials], wait_for_ready: Optional[bool], compression: Optional[grpc.Compression]) -> Tuple[_RPCState, cygrpc.SegregatedCall]:
    deadline = _deadline(timeout)
    state = _RPCState(_STREAM_UNARY_INITIAL_DUE, None, None, None, None)
    initial_metadata_flags = _InitialMetadataFlags().with_wait_for_ready(wait_for_ready)
    augmented_metadata = _compression.augment_metadata(metadata, compression)
    state.rpc_start_time = time.perf_counter()
    state.method = _common.decode(self._method)
    state.target = _common.decode(self._target)
    call = self._channel.segregated_call(cygrpc.PropagationConstants.GRPC_PROPAGATE_DEFAULTS, self._method, None, _determine_deadline(deadline), augmented_metadata, None if credentials is None else credentials._credentials, _stream_unary_invocation_operations_and_tags(augmented_metadata, initial_metadata_flags), self._context)
    _consume_request_iterator(request_iterator, state, call, self._request_serializer, None)
    while True:
        event = call.next_event()
        with state.condition:
            _handle_event(event, state, self._response_deserializer)
            state.condition.notify_all()
            if not state.due:
                break
    return (state, call)