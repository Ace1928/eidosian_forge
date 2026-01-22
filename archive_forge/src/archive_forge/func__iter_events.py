from __future__ import annotations
import json
import inspect
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Iterator, AsyncIterator, cast
from typing_extensions import Self, Protocol, TypeGuard, override, get_origin, runtime_checkable
import httpx
from ._utils import is_mapping, extract_type_var_from_base
from ._exceptions import APIError
def _iter_events(self) -> Iterator[ServerSentEvent]:
    if isinstance(self._decoder, SSEBytesDecoder):
        yield from self._decoder.iter_bytes(self.response.iter_bytes())
    else:
        yield from self._decoder.iter(self.response.iter_lines())