import asyncio
import zlib
from typing import Any, Awaitable, Callable, NamedTuple, Optional, Union  # noqa
from multidict import CIMultiDict
from .abc import AbstractStreamWriter
from .base_protocol import BaseProtocol
from .compression_utils import ZLibCompressor
from .helpers import NO_EXTENSIONS
def _safe_header(string: str) -> str:
    if '\r' in string or '\n' in string:
        raise ValueError('Newline or carriage return detected in headers. Potential header injection attack.')
    return string