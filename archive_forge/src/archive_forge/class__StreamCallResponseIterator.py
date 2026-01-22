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
class _StreamCallResponseIterator:
    _call: Union[_base_call.UnaryStreamCall, _base_call.StreamStreamCall]
    _response_iterator: AsyncIterable[ResponseType]

    def __init__(self, call: Union[_base_call.UnaryStreamCall, _base_call.StreamStreamCall], response_iterator: AsyncIterable[ResponseType]) -> None:
        self._response_iterator = response_iterator
        self._call = call

    def cancel(self) -> bool:
        return self._call.cancel()

    def cancelled(self) -> bool:
        return self._call.cancelled()

    def done(self) -> bool:
        return self._call.done()

    def add_done_callback(self, callback) -> None:
        self._call.add_done_callback(callback)

    def time_remaining(self) -> Optional[float]:
        return self._call.time_remaining()

    async def initial_metadata(self) -> Optional[Metadata]:
        return await self._call.initial_metadata()

    async def trailing_metadata(self) -> Optional[Metadata]:
        return await self._call.trailing_metadata()

    async def code(self) -> grpc.StatusCode:
        return await self._call.code()

    async def details(self) -> str:
        return await self._call.details()

    async def debug_error_string(self) -> Optional[str]:
        return await self._call.debug_error_string()

    def __aiter__(self):
        return self._response_iterator.__aiter__()

    async def wait_for_connection(self) -> None:
        return await self._call.wait_for_connection()