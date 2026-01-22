from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.exceptions import UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.primitives._cipheralgorithm import (
from cryptography.hazmat.primitives.ciphers import algorithms
def _check_aes_key_length(self: Mode, algorithm: CipherAlgorithm) -> None:
    if algorithm.key_size > 256 and algorithm.name == 'AES':
        raise ValueError('Only 128, 192, and 256 bit keys are allowed for this AES mode')