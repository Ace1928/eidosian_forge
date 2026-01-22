from sympy.core import Basic, Integer
import random
def _unrank(k, n):
    if n == 1:
        return str(k % 2)
    m = 2 ** (n - 1)
    if k < m:
        return '0' + _unrank(k, n - 1)
    return '1' + _unrank(m - k % m - 1, n - 1)