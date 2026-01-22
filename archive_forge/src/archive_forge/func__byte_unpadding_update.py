from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.exceptions import AlreadyFinalized
from cryptography.hazmat.bindings._rust import (
def _byte_unpadding_update(buffer_: typing.Optional[bytes], data: bytes, block_size: int) -> typing.Tuple[bytes, bytes]:
    if buffer_ is None:
        raise AlreadyFinalized('Context was already finalized.')
    utils._check_byteslike('data', data)
    buffer_ += bytes(data)
    finished_blocks = max(len(buffer_) // (block_size // 8) - 1, 0)
    result = buffer_[:finished_blocks * (block_size // 8)]
    buffer_ = buffer_[finished_blocks * (block_size // 8):]
    return (buffer_, result)