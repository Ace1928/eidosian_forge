from __future__ import annotations
import abc
import typing
from cryptography.exceptions import (
from cryptography.hazmat.primitives._cipheralgorithm import CipherAlgorithm
from cryptography.hazmat.primitives.ciphers import modes
def _check_limit(self, data_size: int) -> None:
    if self._ctx is None:
        raise AlreadyFinalized('Context was already finalized.')
    self._updated = True
    self._bytes_processed += data_size
    if self._bytes_processed > self._ctx._mode._MAX_ENCRYPTED_BYTES:
        raise ValueError('{} has a maximum encrypted byte limit of {}'.format(self._ctx._mode.name, self._ctx._mode._MAX_ENCRYPTED_BYTES))