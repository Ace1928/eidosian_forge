import asyncio
import enum
import io
import json
import mimetypes
import os
import warnings
from abc import ABC, abstractmethod
from itertools import chain
from typing import (
from multidict import CIMultiDict
from . import hdrs
from .abc import AbstractStreamWriter
from .helpers import (
from .streams import StreamReader
from .typedefs import JSONEncoder, _CIMultiDict
class AsyncIterablePayload(Payload):
    _iter: Optional[_AsyncIterator] = None

    def __init__(self, value: _AsyncIterable, *args: Any, **kwargs: Any) -> None:
        if not isinstance(value, AsyncIterable):
            raise TypeError('value argument must support collections.abc.AsyncIterable interface, got {!r}'.format(type(value)))
        if 'content_type' not in kwargs:
            kwargs['content_type'] = 'application/octet-stream'
        super().__init__(value, *args, **kwargs)
        self._iter = value.__aiter__()

    async def write(self, writer: AbstractStreamWriter) -> None:
        if self._iter:
            try:
                while True:
                    chunk = await self._iter.__anext__()
                    await writer.write(chunk)
            except StopAsyncIteration:
                self._iter = None