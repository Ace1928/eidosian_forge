from __future__ import annotations
from sympy.core.function import Function
from sympy.core.numbers import igcd, igcdex, mod_inverse
from sympy.core.power import isqrt
from sympy.core.singleton import S
from sympy.polys import Poly
from sympy.polys.domains import ZZ
from sympy.polys.galoistools import gf_crt1, gf_crt2, linear_congruence
from .primetest import isprime
from .factor_ import factorint, trailing, totient, multiplicity, perfect_power
from sympy.utilities.misc import as_int
from sympy.core.random import _randint, randint
from itertools import cycle, product
def _sqrt_mod_prime_power(a, p, k):
    """
    Find the solutions to ``x**2 = a mod p**k`` when ``a % p != 0``

    Parameters
    ==========

    a : integer
    p : prime number
    k : positive integer

    Examples
    ========

    >>> from sympy.ntheory.residue_ntheory import _sqrt_mod_prime_power
    >>> _sqrt_mod_prime_power(11, 43, 1)
    [21, 22]

    References
    ==========

    .. [1] P. Hackman "Elementary Number Theory" (2009), page 160
    .. [2] http://www.numbertheory.org/php/squareroot.html
    .. [3] [Gathen99]_
    """
    pk = p ** k
    a = a % pk
    if k == 1:
        if p == 2:
            return [ZZ(a)]
        if not (a % p < 2 or pow(a, (p - 1) // 2, p) == 1):
            return None
        if p % 4 == 3:
            res = pow(a, (p + 1) // 4, p)
        elif p % 8 == 5:
            sign = pow(a, (p - 1) // 4, p)
            if sign == 1:
                res = pow(a, (p + 3) // 8, p)
            else:
                b = pow(4 * a, (p - 5) // 8, p)
                x = 2 * a * b % p
                if pow(x, 2, p) == a:
                    res = x
        else:
            res = _sqrt_mod_tonelli_shanks(a, p)
        return sorted([ZZ(res), ZZ(p - res)])
    if k > 1:
        if p == 2:
            if a % 8 != 1:
                return None
            if k <= 3:
                s = set()
                for i in range(0, pk, 4):
                    s.add(1 + i)
                    s.add(-1 + i)
                return list(s)
            rv = [ZZ(1), ZZ(3), ZZ(5), ZZ(7)]
            n = 3
            res = []
            for r in rv:
                nx = n
                while nx < k:
                    r1 = r ** 2 - a >> nx
                    if r1 % 2:
                        r = r + (1 << nx - 1)
                    nx += 1
                if r not in res:
                    res.append(r)
                x = r + (1 << k - 1)
                if x < 1 << nx and x not in res:
                    if (x ** 2 - a) % pk == 0:
                        res.append(x)
            return res
        rv = _sqrt_mod_prime_power(a, p, 1)
        if not rv:
            return None
        r = rv[0]
        fr = r ** 2 - a
        n = 1
        px = p
        while 1:
            n1 = n
            n1 *= 2
            if n1 > k:
                break
            n = n1
            px = px ** 2
            frinv = igcdex(2 * r, px)[0]
            r = (r - fr * frinv) % px
            fr = r ** 2 - a
        if n < k:
            px = p ** k
            frinv = igcdex(2 * r, px)[0]
            r = (r - fr * frinv) % px
        return [r, px - r]