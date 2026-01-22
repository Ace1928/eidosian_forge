import math
import sys
import struct
from Cryptodome import Random
from Cryptodome.Util.py3compat import iter_range
import struct
import warnings
def _rabinMillerTest(n, rounds, randfunc=None):
    """_rabinMillerTest(n:long, rounds:int, randfunc:callable):int
    Tests if n is prime.
    Returns 0 when n is definitely composite.
    Returns 1 when n is probably prime.
    Returns 2 when n is definitely prime.

    If randfunc is omitted, then Random.new().read is used.

    This function is for internal use only and may be renamed or removed in
    the future.
    """
    if n < 3 or n & 1 == 0:
        return n == 2
    n_1 = n - 1
    b = 0
    m = n_1
    while m & 1 == 0:
        b += 1
        m >>= 1
    tested = []
    for i in iter_range(min(rounds, n - 2)):
        a = getRandomRange(2, n, randfunc)
        while a in tested:
            a = getRandomRange(2, n, randfunc)
        tested.append(a)
        z = pow(a, m, n)
        if z == 1 or z == n_1:
            continue
        composite = 1
        for r in iter_range(b):
            z = z * z % n
            if z == 1:
                return 0
            elif z == n_1:
                composite = 0
                break
        if composite:
            return 0
    return 1