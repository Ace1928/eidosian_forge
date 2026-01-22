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
class _StreamResponseMixin(Call):
    _message_aiter: AsyncIterator[ResponseType]
    _preparation: asyncio.Task
    _response_style: _APIStyle

    def _init_stream_response_mixin(self, preparation: asyncio.Task):
        self._message_aiter = None
        self._preparation = preparation
        self._response_style = _APIStyle.UNKNOWN

    def _update_response_style(self, style: _APIStyle):
        if self._response_style is _APIStyle.UNKNOWN:
            self._response_style = style
        elif self._response_style is not style:
            raise cygrpc.UsageError(_API_STYLE_ERROR)

    def cancel(self) -> bool:
        if super().cancel():
            self._preparation.cancel()
            return True
        else:
            return False

    async def _fetch_stream_responses(self) -> ResponseType:
        message = await self._read()
        while message is not cygrpc.EOF:
            yield message
            message = await self._read()
        await self._raise_for_status()

    def __aiter__(self) -> AsyncIterator[ResponseType]:
        self._update_response_style(_APIStyle.ASYNC_GENERATOR)
        if self._message_aiter is None:
            self._message_aiter = self._fetch_stream_responses()
        return self._message_aiter

    async def _read(self) -> ResponseType:
        await self._preparation
        try:
            raw_response = await self._cython_call.receive_serialized_message()
        except asyncio.CancelledError:
            if not self.cancelled():
                self.cancel()
            raise
        if raw_response is cygrpc.EOF:
            return cygrpc.EOF
        else:
            return _common.deserialize(raw_response, self._response_deserializer)

    async def read(self) -> ResponseType:
        if self.done():
            await self._raise_for_status()
            return cygrpc.EOF
        self._update_response_style(_APIStyle.READER_WRITER)
        response_message = await self._read()
        if response_message is cygrpc.EOF:
            await self._raise_for_status()
        return response_message