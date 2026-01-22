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
class IOBasePayload(Payload):
    _value: IO[Any]

    def __init__(self, value: IO[Any], disposition: str='attachment', *args: Any, **kwargs: Any) -> None:
        if 'filename' not in kwargs:
            kwargs['filename'] = guess_filename(value)
        super().__init__(value, *args, **kwargs)
        if self._filename is not None and disposition is not None:
            if hdrs.CONTENT_DISPOSITION not in self.headers:
                self.set_content_disposition(disposition, filename=self._filename)

    async def write(self, writer: AbstractStreamWriter) -> None:
        loop = asyncio.get_event_loop()
        try:
            chunk = await loop.run_in_executor(None, self._value.read, 2 ** 16)
            while chunk:
                await writer.write(chunk)
                chunk = await loop.run_in_executor(None, self._value.read, 2 ** 16)
        finally:
            await loop.run_in_executor(None, self._value.close)