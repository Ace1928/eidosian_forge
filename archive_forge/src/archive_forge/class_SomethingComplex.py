import random
from mpmath import *
from mpmath.libmp import *
class SomethingComplex:

    def _mpmath_(self, prec, rounding):
        return mp.make_mpc((from_str('1.3', prec, rounding), from_str('1.7', prec, rounding)))