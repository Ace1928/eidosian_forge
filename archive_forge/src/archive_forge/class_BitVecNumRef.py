from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
class BitVecNumRef(BitVecRef):
    """Bit-vector values."""

    def as_long(self):
        """Return a Z3 bit-vector numeral as a Python long (bignum) numeral.

        >>> v = BitVecVal(0xbadc0de, 32)
        >>> v
        195936478
        >>> print("0x%.8x" % v.as_long())
        0x0badc0de
        """
        return int(self.as_string())

    def as_signed_long(self):
        """Return a Z3 bit-vector numeral as a Python long (bignum) numeral.
        The most significant bit is assumed to be the sign.

        >>> BitVecVal(4, 3).as_signed_long()
        -4
        >>> BitVecVal(7, 3).as_signed_long()
        -1
        >>> BitVecVal(3, 3).as_signed_long()
        3
        >>> BitVecVal(2**32 - 1, 32).as_signed_long()
        -1
        >>> BitVecVal(2**64 - 1, 64).as_signed_long()
        -1
        """
        sz = self.size()
        val = self.as_long()
        if val >= 2 ** (sz - 1):
            val = val - 2 ** sz
        if val < -2 ** (sz - 1):
            val = val + 2 ** sz
        return int(val)

    def as_string(self):
        return Z3_get_numeral_string(self.ctx_ref(), self.as_ast())

    def as_binary_string(self):
        return Z3_get_numeral_binary_string(self.ctx_ref(), self.as_ast())