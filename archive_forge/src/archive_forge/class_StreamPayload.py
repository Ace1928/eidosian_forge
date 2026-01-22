import types
import warnings
from typing import Any, Awaitable, Callable, Dict, Tuple
from .abc import AbstractStreamWriter
from .payload import Payload, payload_type
@payload_type(streamer)
class StreamPayload(StreamWrapperPayload):

    def __init__(self, value: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(value(), *args, **kwargs)

    async def write(self, writer: AbstractStreamWriter) -> None:
        await self._value(writer)