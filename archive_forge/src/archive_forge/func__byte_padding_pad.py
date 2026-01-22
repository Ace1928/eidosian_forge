from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.exceptions import AlreadyFinalized
from cryptography.hazmat.bindings._rust import (
def _byte_padding_pad(buffer_: typing.Optional[bytes], block_size: int, paddingfn: typing.Callable[[int], bytes]) -> bytes:
    if buffer_ is None:
        raise AlreadyFinalized('Context was already finalized.')
    pad_size = block_size // 8 - len(buffer_)
    return buffer_ + paddingfn(pad_size)