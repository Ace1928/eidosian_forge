from sympy.core import S, sympify
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import as_int
def _square_and_reduce(u, offset):
    w = [S.Zero] * (2 * len(u) - 1 + offset)
    for i, p in enumerate(u):
        for j, q in enumerate(u):
            w[offset + i + j] += p * q
    for j in range(len(w) - 1, k - 1, -1):
        for i in range(k):
            w[j - i - 1] += w[j] * c[i]
    return w[:k]