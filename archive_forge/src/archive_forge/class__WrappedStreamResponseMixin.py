import asyncio
import functools
from typing import AsyncGenerator, Generic, Iterator, Optional, TypeVar
import grpc
from grpc import aio
from google.api_core import exceptions, grpc_helpers
class _WrappedStreamResponseMixin(Generic[P], _WrappedCall):

    def __init__(self):
        self._wrapped_async_generator = None

    async def read(self) -> P:
        try:
            return await self._call.read()
        except grpc.RpcError as rpc_error:
            raise exceptions.from_grpc_error(rpc_error) from rpc_error

    async def _wrapped_aiter(self) -> AsyncGenerator[P, None]:
        try:
            async for response in self._call:
                yield response
        except grpc.RpcError as rpc_error:
            raise exceptions.from_grpc_error(rpc_error) from rpc_error

    def __aiter__(self) -> AsyncGenerator[P, None]:
        if not self._wrapped_async_generator:
            self._wrapped_async_generator = self._wrapped_aiter()
        return self._wrapped_async_generator