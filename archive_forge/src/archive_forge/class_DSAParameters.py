from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
from cryptography.hazmat.primitives import _serialization, hashes
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
class DSAParameters(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def generate_private_key(self) -> DSAPrivateKey:
        """
        Generates and returns a DSAPrivateKey.
        """

    @abc.abstractmethod
    def parameter_numbers(self) -> DSAParameterNumbers:
        """
        Returns a DSAParameterNumbers.
        """