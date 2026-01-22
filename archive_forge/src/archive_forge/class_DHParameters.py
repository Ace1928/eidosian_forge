from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
from cryptography.hazmat.primitives import _serialization
class DHParameters(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def generate_private_key(self) -> DHPrivateKey:
        """
        Generates and returns a DHPrivateKey.
        """

    @abc.abstractmethod
    def parameter_bytes(self, encoding: _serialization.Encoding, format: _serialization.ParameterFormat) -> bytes:
        """
        Returns the parameters serialized as bytes.
        """

    @abc.abstractmethod
    def parameter_numbers(self) -> DHParameterNumbers:
        """
        Returns a DHParameterNumbers.
        """