from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.exceptions import UnsupportedAlgorithm, _Reasons
from cryptography.hazmat.primitives._cipheralgorithm import (
from cryptography.hazmat.primitives.ciphers import algorithms
class ModeWithNonce(Mode, metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def nonce(self) -> bytes:
        """
        The value of the nonce for this mode as bytes.
        """