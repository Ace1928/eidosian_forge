import types
import warnings
from typing import Any, Awaitable, Callable, Dict, Tuple
from .abc import AbstractStreamWriter
from .payload import Payload, payload_type
class streamer:

    def __init__(self, coro: Callable[..., Awaitable[None]]) -> None:
        warnings.warn('@streamer is deprecated, use async generators instead', DeprecationWarning, stacklevel=2)
        self.coro = coro

    def __call__(self, *args: Any, **kwargs: Any) -> _stream_wrapper:
        return _stream_wrapper(self.coro, args, kwargs)