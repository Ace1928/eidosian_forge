from __future__ import annotations
import abc
import typing
from math import gcd
from cryptography.hazmat.primitives import _serialization, hashes
from cryptography.hazmat.primitives._asymmetric import AsymmetricPadding
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
class RSAPrivateNumbers:

    def __init__(self, p: int, q: int, d: int, dmp1: int, dmq1: int, iqmp: int, public_numbers: RSAPublicNumbers):
        if not isinstance(p, int) or not isinstance(q, int) or (not isinstance(d, int)) or (not isinstance(dmp1, int)) or (not isinstance(dmq1, int)) or (not isinstance(iqmp, int)):
            raise TypeError('RSAPrivateNumbers p, q, d, dmp1, dmq1, iqmp arguments must all be an integers.')
        if not isinstance(public_numbers, RSAPublicNumbers):
            raise TypeError('RSAPrivateNumbers public_numbers must be an RSAPublicNumbers instance.')
        self._p = p
        self._q = q
        self._d = d
        self._dmp1 = dmp1
        self._dmq1 = dmq1
        self._iqmp = iqmp
        self._public_numbers = public_numbers

    @property
    def p(self) -> int:
        return self._p

    @property
    def q(self) -> int:
        return self._q

    @property
    def d(self) -> int:
        return self._d

    @property
    def dmp1(self) -> int:
        return self._dmp1

    @property
    def dmq1(self) -> int:
        return self._dmq1

    @property
    def iqmp(self) -> int:
        return self._iqmp

    @property
    def public_numbers(self) -> RSAPublicNumbers:
        return self._public_numbers

    def private_key(self, backend: typing.Any=None, *, unsafe_skip_rsa_key_validation: bool=False) -> RSAPrivateKey:
        from cryptography.hazmat.backends.openssl.backend import backend as ossl
        return ossl.load_rsa_private_numbers(self, unsafe_skip_rsa_key_validation)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RSAPrivateNumbers):
            return NotImplemented
        return self.p == other.p and self.q == other.q and (self.d == other.d) and (self.dmp1 == other.dmp1) and (self.dmq1 == other.dmq1) and (self.iqmp == other.iqmp) and (self.public_numbers == other.public_numbers)

    def __hash__(self) -> int:
        return hash((self.p, self.q, self.d, self.dmp1, self.dmq1, self.iqmp, self.public_numbers))