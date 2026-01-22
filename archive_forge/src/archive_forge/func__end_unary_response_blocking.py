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
def _end_unary_response_blocking(state: _RPCState, call: cygrpc.SegregatedCall, with_call: bool, deadline: Optional[float]) -> Union[ResponseType, Tuple[ResponseType, grpc.Call]]:
    if state.code is grpc.StatusCode.OK:
        if with_call:
            rendezvous = _MultiThreadedRendezvous(state, call, None, deadline)
            return (state.response, rendezvous)
        else:
            return state.response
    else:
        raise _InactiveRpcError(state)