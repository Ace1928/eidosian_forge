import operator
import math
import cmath
def _sinpi_complex(z):
    if z.real < 0:
        return -_sinpi_complex(-z)
    n, r = divmod(z.real, 0.5)
    z = pi * complex(r, z.imag)
    n %= 4
    if n == 0:
        return cmath.sin(z)
    if n == 1:
        return cmath.cos(z)
    if n == 2:
        return -cmath.sin(z)
    if n == 3:
        return -cmath.cos(z)