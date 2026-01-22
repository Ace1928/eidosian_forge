import types
import warnings
from typing import Any, Awaitable, Callable, Dict, Tuple
from .abc import AbstractStreamWriter
from .payload import Payload, payload_type
@payload_type(_stream_wrapper)
class StreamWrapperPayload(Payload):

    async def write(self, writer: AbstractStreamWriter) -> None:
        await self._value(writer)