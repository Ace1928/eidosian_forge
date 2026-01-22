from sympy.core.sympify import sympify
from sympy.holonomic.holonomic import DMFsubs
from mpmath import mp
def _rk4(red, x0, x1, y0, a):
    """
    Runge-Kutta 4th order numerical method.
    """
    A = sympify(x0)._to_mpmath(mp.prec)
    B = sympify(x1)._to_mpmath(mp.prec)
    y_0 = [sympify(i)._to_mpmath(mp.prec) for i in y0]
    h = B - A
    f_0_n = 0
    f_1_n = 0
    f_2_n = 0
    f_3_n = 0
    f_0 = y_0[1:]
    for i in range(a):
        f_0_n += sympify(DMFsubs(red[i], A, mpm=True))._to_mpmath(mp.prec) * y_0[i]
    f_0.append(f_0_n)
    f_1 = [y_0[i] + f_0[i] * h / 2 for i in range(1, a)]
    for i in range(a):
        f_1_n += sympify(DMFsubs(red[i], A + h / 2, mpm=True))._to_mpmath(mp.prec) * (y_0[i] + f_0[i] * h / 2)
    f_1.append(f_1_n)
    f_2 = [y_0[i] + f_1[i] * h / 2 for i in range(1, a)]
    for i in range(a):
        f_2_n += sympify(DMFsubs(red[i], A + h / 2, mpm=True))._to_mpmath(mp.prec) * (y_0[i] + f_1[i] * h / 2)
    f_2.append(f_2_n)
    f_3 = [y_0[i] + f_2[i] * h for i in range(1, a)]
    for i in range(a):
        f_3_n += sympify(DMFsubs(red[i], A + h, mpm=True))._to_mpmath(mp.prec) * (y_0[i] + f_2[i] * h)
    f_3.append(f_3_n)
    sol = []
    for i in range(a):
        sol.append(y_0[i] + h * (f_0[i] + 2 * f_1[i] + 2 * f_2[i] + f_3[i]) / 6)
    return sol