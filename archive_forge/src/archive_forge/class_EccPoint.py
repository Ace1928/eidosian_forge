from __future__ import print_function
import re
import struct
import binascii
from collections import namedtuple
from Cryptodome.Util.py3compat import bord, tobytes, tostr, bchr, is_string
from Cryptodome.Util.number import bytes_to_long, long_to_bytes
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Util.asn1 import (DerObjectId, DerOctetString, DerSequence,
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib, VoidPointer,
from Cryptodome.PublicKey import (_expand_subject_public_key_info,
from Cryptodome.Hash import SHA512, SHAKE256
from Cryptodome.Random import get_random_bytes
from Cryptodome.Random.random import getrandbits
class EccPoint(object):
    """A class to model a point on an Elliptic Curve.

    The class supports operators for:

    * Adding two points: ``R = S + T``
    * In-place addition: ``S += T``
    * Negating a point: ``R = -T``
    * Comparing two points: ``if S == T: ...`` or ``if S != T: ...``
    * Multiplying a point by a scalar: ``R = S*k``
    * In-place multiplication by a scalar: ``T *= k``

    :ivar x: The affine X-coordinate of the ECC point
    :vartype x: integer

    :ivar y: The affine Y-coordinate of the ECC point
    :vartype y: integer

    :ivar xy: The tuple with affine X- and Y- coordinates
    """

    def __init__(self, x, y, curve='p256'):
        try:
            self._curve = _curves[curve]
        except KeyError:
            raise ValueError('Unknown curve name %s' % str(curve))
        self._curve_name = curve
        modulus_bytes = self.size_in_bytes()
        xb = long_to_bytes(x, modulus_bytes)
        yb = long_to_bytes(y, modulus_bytes)
        if len(xb) != modulus_bytes or len(yb) != modulus_bytes:
            raise ValueError('Incorrect coordinate length')
        new_point = lib_func(self, 'new_point')
        free_func = lib_func(self, 'free_point')
        self._point = VoidPointer()
        try:
            context = self._curve.context.get()
        except AttributeError:
            context = null_pointer
        result = new_point(self._point.address_of(), c_uint8_ptr(xb), c_uint8_ptr(yb), c_size_t(modulus_bytes), context)
        if result:
            if result == 15:
                raise ValueError('The EC point does not belong to the curve')
            raise ValueError('Error %d while instantiating an EC point' % result)
        self._point = SmartPointer(self._point.get(), free_func)

    def set(self, point):
        clone = lib_func(self, 'clone')
        free_func = lib_func(self, 'free_point')
        self._point = VoidPointer()
        result = clone(self._point.address_of(), point._point.get())
        if result:
            raise ValueError('Error %d while cloning an EC point' % result)
        self._point = SmartPointer(self._point.get(), free_func)
        return self

    def __eq__(self, point):
        if not isinstance(point, EccPoint):
            return False
        cmp_func = lib_func(self, 'cmp')
        return 0 == cmp_func(self._point.get(), point._point.get())

    def __ne__(self, point):
        return not self == point

    def __neg__(self):
        neg_func = lib_func(self, 'neg')
        np = self.copy()
        result = neg_func(np._point.get())
        if result:
            raise ValueError('Error %d while inverting an EC point' % result)
        return np

    def copy(self):
        """Return a copy of this point."""
        x, y = self.xy
        np = EccPoint(x, y, self._curve_name)
        return np

    def _is_eddsa(self):
        return self._curve.name in ('ed25519', 'ed448')

    def is_point_at_infinity(self):
        """``True`` if this is the *point-at-infinity*."""
        if self._is_eddsa():
            return self.x == 0
        else:
            return self.xy == (0, 0)

    def point_at_infinity(self):
        """Return the *point-at-infinity* for the curve."""
        if self._is_eddsa():
            return EccPoint(0, 1, self._curve_name)
        else:
            return EccPoint(0, 0, self._curve_name)

    @property
    def x(self):
        return self.xy[0]

    @property
    def y(self):
        return self.xy[1]

    @property
    def xy(self):
        modulus_bytes = self.size_in_bytes()
        xb = bytearray(modulus_bytes)
        yb = bytearray(modulus_bytes)
        get_xy = lib_func(self, 'get_xy')
        result = get_xy(c_uint8_ptr(xb), c_uint8_ptr(yb), c_size_t(modulus_bytes), self._point.get())
        if result:
            raise ValueError('Error %d while encoding an EC point' % result)
        return (Integer(bytes_to_long(xb)), Integer(bytes_to_long(yb)))

    def size_in_bytes(self):
        """Size of each coordinate, in bytes."""
        return (self.size_in_bits() + 7) // 8

    def size_in_bits(self):
        """Size of each coordinate, in bits."""
        return self._curve.modulus_bits

    def double(self):
        """Double this point (in-place operation).

        Returns:
            This same object (to enable chaining).
        """
        double_func = lib_func(self, 'double')
        result = double_func(self._point.get())
        if result:
            raise ValueError('Error %d while doubling an EC point' % result)
        return self

    def __iadd__(self, point):
        """Add a second point to this one"""
        add_func = lib_func(self, 'add')
        result = add_func(self._point.get(), point._point.get())
        if result:
            if result == 16:
                raise ValueError('EC points are not on the same curve')
            raise ValueError('Error %d while adding two EC points' % result)
        return self

    def __add__(self, point):
        """Return a new point, the addition of this one and another"""
        np = self.copy()
        np += point
        return np

    def __imul__(self, scalar):
        """Multiply this point by a scalar"""
        scalar_func = lib_func(self, 'scalar')
        if scalar < 0:
            raise ValueError('Scalar multiplication is only defined for non-negative integers')
        sb = long_to_bytes(scalar)
        result = scalar_func(self._point.get(), c_uint8_ptr(sb), c_size_t(len(sb)), c_ulonglong(getrandbits(64)))
        if result:
            raise ValueError('Error %d during scalar multiplication' % result)
        return self

    def __mul__(self, scalar):
        """Return a new point, the scalar product of this one"""
        np = self.copy()
        np *= scalar
        return np

    def __rmul__(self, left_hand):
        return self.__mul__(left_hand)