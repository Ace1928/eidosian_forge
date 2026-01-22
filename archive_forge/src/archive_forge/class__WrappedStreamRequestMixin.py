import asyncio
import functools
from typing import AsyncGenerator, Generic, Iterator, Optional, TypeVar
import grpc
from grpc import aio
from google.api_core import exceptions, grpc_helpers
class _WrappedStreamRequestMixin(_WrappedCall):

    async def write(self, request):
        try:
            await self._call.write(request)
        except grpc.RpcError as rpc_error:
            raise exceptions.from_grpc_error(rpc_error) from rpc_error

    async def done_writing(self):
        try:
            await self._call.done_writing()
        except grpc.RpcError as rpc_error:
            raise exceptions.from_grpc_error(rpc_error) from rpc_error