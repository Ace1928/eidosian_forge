import os
from tempfile import TemporaryDirectory
from sympy.concrete.summations import Sum
from sympy.core.numbers import (I, oo, pi)
from sympy.core.relational import Ne
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (LambertW, exp, exp_polar, log)
from sympy.functions.elementary.miscellaneous import (real_root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.hyper import meijerg
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import And
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.external import import_module
from sympy.plotting.plot import (
from sympy.plotting.plot import (
from sympy.testing.pytest import skip, raises, warns, warns_deprecated_sympy
from sympy.utilities import lambdify as lambdify_
from sympy.utilities.exceptions import ignore_warnings
def test_plot_and_save_5():
    if not matplotlib:
        skip('Matplotlib not the default backend')
    x = Symbol('x')
    y = Symbol('y')
    with TemporaryDirectory(prefix='sympy_') as tmpdir:
        s = Sum(1 / x ** y, (x, 1, oo))
        p = plot(s, (y, 2, 10))
        filename = 'test_advanced_inf_sum.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p = plot(Sum(1 / x, (x, 1, y)), (y, 2, 10), show=False)
        p[0].only_integers = True
        p[0].steps = True
        filename = 'test_advanced_fin_sum.png'
        with ignore_warnings(UserWarning):
            p.save(os.path.join(tmpdir, filename))
        p._backend.close()