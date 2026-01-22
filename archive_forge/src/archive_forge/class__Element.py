from Cryptodome.Util.py3compat import is_native_int
from Cryptodome.Util import number
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Random import get_random_bytes as rng
class _Element(object):
    """Element of GF(2^128) field"""
    irr_poly = 1 + 2 + 4 + 128 + 2 ** 128

    def __init__(self, encoded_value):
        """Initialize the element to a certain value.

        The value passed as parameter is internally encoded as
        a 128-bit integer, where each bit represents a polynomial
        coefficient. The LSB is the constant coefficient.
        """
        if is_native_int(encoded_value):
            self._value = encoded_value
        elif len(encoded_value) == 16:
            self._value = bytes_to_long(encoded_value)
        else:
            raise ValueError('The encoded value must be an integer or a 16 byte string')

    def __eq__(self, other):
        return self._value == other._value

    def __int__(self):
        """Return the field element, encoded as a 128-bit integer."""
        return self._value

    def encode(self):
        """Return the field element, encoded as a 16 byte string."""
        return long_to_bytes(self._value, 16)

    def __mul__(self, factor):
        f1 = self._value
        f2 = factor._value
        if f2 > f1:
            f1, f2 = (f2, f1)
        if self.irr_poly in (f1, f2):
            return _Element(0)
        mask1 = 2 ** 128
        v, z = (f1, 0)
        while f2:
            mask2 = int(bin(f2 & 1)[2:] * 128, base=2)
            z = mask2 & (z ^ v) | mask1 - mask2 - 1 & z
            v <<= 1
            mask3 = int(bin(v >> 128 & 1)[2:] * 128, base=2)
            v = mask3 & (v ^ self.irr_poly) | mask1 - mask3 - 1 & v
            f2 >>= 1
        return _Element(z)

    def __add__(self, term):
        return _Element(self._value ^ term._value)

    def inverse(self):
        """Return the inverse of this element in GF(2^128)."""
        if self._value == 0:
            raise ValueError('Inversion of zero')
        r0, r1 = (self._value, self.irr_poly)
        s0, s1 = (1, 0)
        while r1 > 0:
            q = _div_gf2(r0, r1)[0]
            r0, r1 = (r1, r0 ^ _mult_gf2(q, r1))
            s0, s1 = (s1, s0 ^ _mult_gf2(q, s1))
        return _Element(s0)

    def __pow__(self, exponent):
        result = _Element(self._value)
        for _ in range(exponent - 1):
            result = result * self
        return result