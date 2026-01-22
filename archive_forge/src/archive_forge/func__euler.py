from sympy.core.sympify import sympify
from sympy.holonomic.holonomic import DMFsubs
from mpmath import mp
def _euler(red, x0, x1, y0, a):
    """
    Euler's method for numerical integration.
    From x0 to x1 with initial values given at x0 as vector y0.
    """
    A = sympify(x0)._to_mpmath(mp.prec)
    B = sympify(x1)._to_mpmath(mp.prec)
    y_0 = [sympify(i)._to_mpmath(mp.prec) for i in y0]
    h = B - A
    f_0 = y_0[1:]
    f_0_n = 0
    for i in range(a):
        f_0_n += sympify(DMFsubs(red[i], A, mpm=True))._to_mpmath(mp.prec) * y_0[i]
    f_0.append(f_0_n)
    sol = []
    for i in range(a):
        sol.append(y_0[i] + h * f_0[i])
    return sol