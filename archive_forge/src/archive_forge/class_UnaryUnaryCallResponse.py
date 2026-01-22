from abc import ABCMeta
from abc import abstractmethod
import asyncio
import collections
import functools
from typing import (
import grpc
from grpc._cython import cygrpc
from . import _base_call
from ._call import AioRpcError
from ._call import StreamStreamCall
from ._call import StreamUnaryCall
from ._call import UnaryStreamCall
from ._call import UnaryUnaryCall
from ._call import _API_STYLE_ERROR
from ._call import _RPC_ALREADY_FINISHED_DETAILS
from ._call import _RPC_HALF_CLOSED_DETAILS
from ._metadata import Metadata
from ._typing import DeserializingFunction
from ._typing import DoneCallbackType
from ._typing import RequestIterableType
from ._typing import RequestType
from ._typing import ResponseIterableType
from ._typing import ResponseType
from ._typing import SerializingFunction
from ._utils import _timeout_to_deadline
class UnaryUnaryCallResponse(_base_call.UnaryUnaryCall):
    """Final UnaryUnaryCall class finished with a response."""
    _response: ResponseType

    def __init__(self, response: ResponseType) -> None:
        self._response = response

    def cancel(self) -> bool:
        return False

    def cancelled(self) -> bool:
        return False

    def done(self) -> bool:
        return True

    def add_done_callback(self, unused_callback) -> None:
        raise NotImplementedError()

    def time_remaining(self) -> Optional[float]:
        raise NotImplementedError()

    async def initial_metadata(self) -> Optional[Metadata]:
        return None

    async def trailing_metadata(self) -> Optional[Metadata]:
        return None

    async def code(self) -> grpc.StatusCode:
        return grpc.StatusCode.OK

    async def details(self) -> str:
        return ''

    async def debug_error_string(self) -> Optional[str]:
        return None

    def __await__(self):
        if False:
            yield None
        return self._response

    async def wait_for_connection(self) -> None:
        pass