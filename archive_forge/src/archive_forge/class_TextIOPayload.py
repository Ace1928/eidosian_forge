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
class TextIOPayload(IOBasePayload):
    _value: TextIO

    def __init__(self, value: TextIO, *args: Any, encoding: Optional[str]=None, content_type: Optional[str]=None, **kwargs: Any) -> None:
        if encoding is None:
            if content_type is None:
                encoding = 'utf-8'
                content_type = 'text/plain; charset=utf-8'
            else:
                mimetype = parse_mimetype(content_type)
                encoding = mimetype.parameters.get('charset', 'utf-8')
        elif content_type is None:
            content_type = 'text/plain; charset=%s' % encoding
        super().__init__(value, *args, content_type=content_type, encoding=encoding, **kwargs)

    @property
    def size(self) -> Optional[int]:
        try:
            return os.fstat(self._value.fileno()).st_size - self._value.tell()
        except OSError:
            return None

    async def write(self, writer: AbstractStreamWriter) -> None:
        loop = asyncio.get_event_loop()
        try:
            chunk = await loop.run_in_executor(None, self._value.read, 2 ** 16)
            while chunk:
                data = chunk.encode(encoding=self._encoding) if self._encoding else chunk.encode()
                await writer.write(data)
                chunk = await loop.run_in_executor(None, self._value.read, 2 ** 16)
        finally:
            await loop.run_in_executor(None, self._value.close)