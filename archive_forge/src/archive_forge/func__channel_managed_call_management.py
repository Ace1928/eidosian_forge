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
def _channel_managed_call_management(state: _ChannelCallState):

    def create(flags: int, method: bytes, host: Optional[str], deadline: Optional[float], metadata: Optional[MetadataType], credentials: Optional[cygrpc.CallCredentials], operations: Sequence[Sequence[cygrpc.Operation]], event_handler: UserTag, context) -> cygrpc.IntegratedCall:
        """Creates a cygrpc.IntegratedCall.

        Args:
          flags: An integer bitfield of call flags.
          method: The RPC method.
          host: A host string for the created call.
          deadline: A float to be the deadline of the created call or None if
            the call is to have an infinite deadline.
          metadata: The metadata for the call or None.
          credentials: A cygrpc.CallCredentials or None.
          operations: A sequence of sequences of cygrpc.Operations to be
            started on the call.
          event_handler: A behavior to call to handle the events resultant from
            the operations on the call.
          context: Context object for distributed tracing.
        Returns:
          A cygrpc.IntegratedCall with which to conduct an RPC.
        """
        operations_and_tags = tuple(((operation, event_handler) for operation in operations))
        with state.lock:
            call = state.channel.integrated_call(flags, method, host, deadline, metadata, credentials, operations_and_tags, context)
            if state.managed_calls == 0:
                state.managed_calls = 1
                _run_channel_spin_thread(state)
            else:
                state.managed_calls += 1
            return call
    return create