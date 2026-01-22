from __future__ import annotations
import abc
import typing
from cryptography.hazmat.bindings._rust import openssl as rust_openssl
from cryptography.hazmat.primitives import _serialization, hashes
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
class DSAPrivateNumbers:

    def __init__(self, x: int, public_numbers: DSAPublicNumbers):
        if not isinstance(x, int):
            raise TypeError('DSAPrivateNumbers x argument must be an integer.')
        if not isinstance(public_numbers, DSAPublicNumbers):
            raise TypeError('public_numbers must be a DSAPublicNumbers instance.')
        self._public_numbers = public_numbers
        self._x = x

    @property
    def x(self) -> int:
        return self._x

    @property
    def public_numbers(self) -> DSAPublicNumbers:
        return self._public_numbers

    def private_key(self, backend: typing.Any=None) -> DSAPrivateKey:
        from cryptography.hazmat.backends.openssl.backend import backend as ossl
        return ossl.load_dsa_private_numbers(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DSAPrivateNumbers):
            return NotImplemented
        return self.x == other.x and self.public_numbers == other.public_numbers