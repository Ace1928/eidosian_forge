from __future__ import annotations
import abc
import typing
from cryptography.exceptions import (
from cryptography.hazmat.primitives._cipheralgorithm import CipherAlgorithm
from cryptography.hazmat.primitives.ciphers import modes
class AEADEncryptionContext(AEADCipherContext, metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def tag(self) -> bytes:
        """
        Returns tag bytes. This is only available after encryption is
        finalized.
        """