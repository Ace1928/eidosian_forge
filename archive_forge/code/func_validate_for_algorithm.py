from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.exceptions import UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.primitives._cipheralgorithm import (
from cryptography.hazmat.primitives.ciphers import algorithms
def validate_for_algorithm(self, algorithm: CipherAlgorithm) -> None:
    _check_aes_key_length(self, algorithm)
    if not isinstance(algorithm, BlockCipherAlgorithm):
        raise UnsupportedAlgorithm('GCM requires a block cipher algorithm', _Reasons.UNSUPPORTED_CIPHER)
    block_size_bytes = algorithm.block_size // 8
    if self._tag is not None and len(self._tag) > block_size_bytes:
        raise ValueError('Authentication tag cannot be more than {} bytes.'.format(block_size_bytes))