from sympy.core.sympify import sympify
from sympy.holonomic.holonomic import DMFsubs
from mpmath import mp
def _evalf(func, points, derivatives=False, method='RK4'):
    """
    Numerical methods for numerical integration along a given set of
    points in the complex plane.
    """
    ann = func.annihilator
    a = ann.order
    R = ann.parent.base
    K = R.get_field()
    if method == 'Euler':
        meth = _euler
    else:
        meth = _rk4
    dmf = []
    for j in ann.listofpoly:
        dmf.append(K.new(j.rep))
    red = [-dmf[i] / dmf[a] for i in range(a)]
    y0 = func.y0
    if len(y0) < a:
        raise TypeError('Not Enough Initial Conditions')
    x0 = func.x0
    sol = [meth(red, x0, points[0], y0, a)]
    for i, j in enumerate(points[1:]):
        sol.append(meth(red, points[i], j, sol[-1], a))
    if not derivatives:
        return [sympify(i[0]) for i in sol]
    else:
        return sympify(sol)