from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.exceptions import UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.primitives._cipheralgorithm import (
from cryptography.hazmat.primitives.ciphers import algorithms
class CBC(ModeWithInitializationVector):
    name = 'CBC'

    def __init__(self, initialization_vector: bytes):
        utils._check_byteslike('initialization_vector', initialization_vector)
        self._initialization_vector = initialization_vector

    @property
    def initialization_vector(self) -> bytes:
        return self._initialization_vector
    validate_for_algorithm = _check_iv_and_key_length