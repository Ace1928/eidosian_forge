from __future__ import annotations
import io
import typing as t
from functools import partial
from functools import update_wrapper
from .exceptions import ClientDisconnected
from .exceptions import RequestEntityTooLarge
from .sansio import utils as _sansio_utils
from .sansio.utils import host_is_trusted  # noqa: F401 # Imported as part of API
def _first_iteration(self) -> tuple[bytes | None, int]:
    chunk = None
    if self.seekable:
        self.iterable.seek(self.start_byte)
        self.read_length = self.iterable.tell()
        contextual_read_length = self.read_length
    else:
        while self.read_length <= self.start_byte:
            chunk = self._next_chunk()
        if chunk is not None:
            chunk = chunk[self.start_byte - self.read_length:]
        contextual_read_length = self.start_byte
    return (chunk, contextual_read_length)