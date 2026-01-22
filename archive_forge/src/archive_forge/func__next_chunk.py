from __future__ import annotations
import io
import typing as t
from functools import partial
from functools import update_wrapper
from .exceptions import ClientDisconnected
from .exceptions import RequestEntityTooLarge
from .sansio import utils as _sansio_utils
from .sansio.utils import host_is_trusted  # noqa: F401 # Imported as part of API
def _next_chunk(self) -> bytes:
    try:
        chunk = next(self.iterable)
        self.read_length += len(chunk)
        return chunk
    except StopIteration:
        self.end_reached = True
        raise