import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
@constant_memo
def apery_fixed(prec):
    prec += 20
    d = MPZ_ONE << prec
    term = MPZ(77) << prec
    n = 1
    s = MPZ_ZERO
    while term:
        s += term
        d *= n ** 10
        d //= (2 * n + 1) ** 5 * (2 * n) ** 5
        term = (-1) ** n * (205 * n ** 2 + 250 * n + 77) * d
        n += 1
    return s >> 20 + 6