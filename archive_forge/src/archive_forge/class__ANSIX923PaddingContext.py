from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.exceptions import AlreadyFinalized
from cryptography.hazmat.bindings._rust import (
class _ANSIX923PaddingContext(PaddingContext):
    _buffer: typing.Optional[bytes]

    def __init__(self, block_size: int):
        self.block_size = block_size
        self._buffer = b''

    def update(self, data: bytes) -> bytes:
        self._buffer, result = _byte_padding_update(self._buffer, data, self.block_size)
        return result

    def _padding(self, size: int) -> bytes:
        return bytes([0]) * (size - 1) + bytes([size])

    def finalize(self) -> bytes:
        result = _byte_padding_pad(self._buffer, self.block_size, self._padding)
        self._buffer = None
        return result