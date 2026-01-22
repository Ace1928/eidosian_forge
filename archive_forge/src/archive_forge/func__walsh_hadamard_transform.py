from sympy.core import S, Symbol, sympify
from sympy.core.function import expand_mul
from sympy.core.numbers import pi, I
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.ntheory import isprime, primitive_root
from sympy.utilities.iterables import ibin, iterable
from sympy.utilities.misc import as_int
def _walsh_hadamard_transform(seq, inverse=False):
    """Utility function for the Walsh Hadamard Transform"""
    if not iterable(seq):
        raise TypeError('Expected a sequence of coefficients for Walsh Hadamard Transform')
    a = [sympify(arg) for arg in seq]
    n = len(a)
    if n < 2:
        return a
    if n & n - 1:
        n = 2 ** n.bit_length()
    a += [S.Zero] * (n - len(a))
    h = 2
    while h <= n:
        hf = h // 2
        for i in range(0, n, h):
            for j in range(hf):
                u, v = (a[i + j], a[i + j + hf])
                a[i + j], a[i + j + hf] = (u + v, u - v)
        h *= 2
    if inverse:
        a = [x / n for x in a]
    return a