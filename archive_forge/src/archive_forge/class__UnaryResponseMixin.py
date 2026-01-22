import asyncio
import enum
from functools import partial
import inspect
import logging
import traceback
from typing import Any, AsyncIterator, Generator, Generic, Optional, Tuple
import grpc
from grpc import _common
from grpc._cython import cygrpc
from . import _base_call
from ._metadata import Metadata
from ._typing import DeserializingFunction
from ._typing import DoneCallbackType
from ._typing import MetadatumType
from ._typing import RequestIterableType
from ._typing import RequestType
from ._typing import ResponseType
from ._typing import SerializingFunction
class _UnaryResponseMixin(Call, Generic[ResponseType]):
    _call_response: asyncio.Task

    def _init_unary_response_mixin(self, response_task: asyncio.Task):
        self._call_response = response_task

    def cancel(self) -> bool:
        if super().cancel():
            self._call_response.cancel()
            return True
        else:
            return False

    def __await__(self) -> Generator[Any, None, ResponseType]:
        """Wait till the ongoing RPC request finishes."""
        try:
            response = (yield from self._call_response)
        except asyncio.CancelledError:
            if not self.cancelled():
                self.cancel()
            raise
        if response is cygrpc.EOF:
            if self._cython_call.is_locally_cancelled():
                raise asyncio.CancelledError()
            else:
                raise _create_rpc_error(self._cython_call._initial_metadata, self._cython_call._status)
        else:
            return response