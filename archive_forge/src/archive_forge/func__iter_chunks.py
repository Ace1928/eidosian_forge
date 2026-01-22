from __future__ import annotations
import json
import inspect
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, TypeVar, Iterator, AsyncIterator, cast
from typing_extensions import Self, Protocol, TypeGuard, override, get_origin, runtime_checkable
import httpx
from ._utils import is_mapping, extract_type_var_from_base
from ._exceptions import APIError
def _iter_chunks(self, iterator: Iterator[bytes]) -> Iterator[bytes]:
    """Given an iterator that yields raw binary data, iterate over it and yield individual SSE chunks"""
    data = b''
    for chunk in iterator:
        for line in chunk.splitlines(keepends=True):
            data += line
            if data.endswith((b'\r\r', b'\n\n', b'\r\n\r\n')):
                yield data
                data = b''
    if data:
        yield data