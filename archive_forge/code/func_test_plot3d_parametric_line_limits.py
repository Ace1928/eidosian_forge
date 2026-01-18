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
def test_plot3d_parametric_line_limits():
    if not matplotlib:
        skip('Matplotlib not the default backend')
    x = Symbol('x')
    v1 = (2 * cos(x), 2 * sin(x), 2 * x, (x, -5, 5))
    v2 = (sin(x), cos(x), x, (x, -5, 5))
    p = plot3d_parametric_line(v1, v2)
    backend = p._backend
    xmin, xmax = backend.ax[0].get_xlim()
    assert abs(xmin + 2) < 0.01
    assert abs(xmax - 2) < 0.01
    ymin, ymax = backend.ax[0].get_ylim()
    assert abs(ymin + 2) < 0.01
    assert abs(ymax - 2) < 0.01
    zmin, zmax = backend.ax[0].get_zlim()
    assert abs(zmin + 10) < 0.01
    assert abs(zmax - 10) < 0.01
    p = plot3d_parametric_line(v2, v1)
    backend = p._backend
    xmin, xmax = backend.ax[0].get_xlim()
    assert abs(xmin + 2) < 0.01
    assert abs(xmax - 2) < 0.01
    ymin, ymax = backend.ax[0].get_ylim()
    assert abs(ymin + 2) < 0.01
    assert abs(ymax - 2) < 0.01
    zmin, zmax = backend.ax[0].get_zlim()
    assert abs(zmin + 10) < 0.01
    assert abs(zmax - 10) < 0.01