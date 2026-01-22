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
class BytesPayload(Payload):

    def __init__(self, value: ByteString, *args: Any, **kwargs: Any) -> None:
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise TypeError(f'value argument must be byte-ish, not {type(value)!r}')
        if 'content_type' not in kwargs:
            kwargs['content_type'] = 'application/octet-stream'
        super().__init__(value, *args, **kwargs)
        if isinstance(value, memoryview):
            self._size = value.nbytes
        else:
            self._size = len(value)
        if self._size > TOO_LARGE_BYTES_BODY:
            kwargs = {'source': self}
            warnings.warn('Sending a large body directly with raw bytes might lock the event loop. You should probably pass an io.BytesIO object instead', ResourceWarning, **kwargs)

    async def write(self, writer: AbstractStreamWriter) -> None:
        await writer.write(self._value)