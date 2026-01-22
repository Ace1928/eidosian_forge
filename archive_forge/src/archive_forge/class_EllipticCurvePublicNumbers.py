from __future__ import annotations
import abc
import typing
from cryptography import utils
from cryptography.hazmat._oid import ObjectIdentifier
from cryptography.hazmat.primitives import _serialization, hashes
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils
class EllipticCurvePublicNumbers:

    def __init__(self, x: int, y: int, curve: EllipticCurve):
        if not isinstance(x, int) or not isinstance(y, int):
            raise TypeError('x and y must be integers.')
        if not isinstance(curve, EllipticCurve):
            raise TypeError('curve must provide the EllipticCurve interface.')
        self._y = y
        self._x = x
        self._curve = curve

    def public_key(self, backend: typing.Any=None) -> EllipticCurvePublicKey:
        from cryptography.hazmat.backends.openssl.backend import backend as ossl
        return ossl.load_elliptic_curve_public_numbers(self)

    @property
    def curve(self) -> EllipticCurve:
        return self._curve

    @property
    def x(self) -> int:
        return self._x

    @property
    def y(self) -> int:
        return self._y

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EllipticCurvePublicNumbers):
            return NotImplemented
        return self.x == other.x and self.y == other.y and (self.curve.name == other.curve.name) and (self.curve.key_size == other.curve.key_size)

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.curve.name, self.curve.key_size))

    def __repr__(self) -> str:
        return '<EllipticCurvePublicNumbers(curve={0.curve.name}, x={0.x}, y={0.y}>'.format(self)