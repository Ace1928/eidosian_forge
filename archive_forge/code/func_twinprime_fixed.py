import math
import sys
from .backend import xrange
from .backend import MPZ, MPZ_ZERO, MPZ_ONE, MPZ_THREE, gmpy
from .libintmath import list_primes, ifac, ifac2, moebius
from .libmpf import (\
from .libelefun import (\
from .libmpc import (\
@constant_memo
def twinprime_fixed(prec):

    def I(n):
        return sum((moebius(d) << n // d for d in xrange(1, n + 1) if not n % d)) // n
    wp = 2 * prec + 30
    res = fone
    primes = [from_rational(1, p, wp) for p in [2, 3, 5, 7]]
    ppowers = [mpf_mul(p, p, wp) for p in primes]
    n = 2
    while 1:
        a = mpf_zeta_int(n, wp)
        for i in range(4):
            a = mpf_mul(a, mpf_sub(fone, ppowers[i]), wp)
            ppowers[i] = mpf_mul(ppowers[i], primes[i], wp)
        a = mpf_pow_int(a, -I(n), wp)
        if mpf_pos(a, prec + 10, 'n') == fone:
            break
        res = mpf_mul(res, a, wp)
        n += 1
    res = mpf_mul(res, from_int(3 * 15 * 35), wp)
    res = mpf_div(res, from_int(4 * 16 * 36), wp)
    return to_fixed(res, prec)