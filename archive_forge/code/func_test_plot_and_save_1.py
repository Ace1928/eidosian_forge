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
def test_plot_and_save_1():
    if not matplotlib:
        skip('Matplotlib not the default backend')
    x = Symbol('x')
    y = Symbol('y')
    with TemporaryDirectory(prefix='sympy_') as tmpdir:
        p = plot(x, legend=True, label='f1')
        p = plot(x * sin(x), x * cos(x), label='f2')
        p.extend(p)
        p[0].line_color = lambda a: a
        p[1].line_color = 'b'
        p.title = 'Big title'
        p.xlabel = 'the x axis'
        p[1].label = 'straight line'
        p.legend = True
        p.aspect_ratio = (1, 1)
        p.xlim = (-15, 20)
        filename = 'test_basic_options_and_colors.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p.extend(plot(x + 1))
        p.append(plot(x + 3, x ** 2)[1])
        filename = 'test_plot_extend_append.png'
        p.save(os.path.join(tmpdir, filename))
        p[2] = plot(x ** 2, (x, -2, 3))
        filename = 'test_plot_setitem.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p = plot(sin(x), (x, -2 * pi, 4 * pi))
        filename = 'test_line_explicit.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p = plot(sin(x))
        filename = 'test_line_default_range.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p = plot((x ** 2, (x, -5, 5)), (x ** 3, (x, -3, 3)))
        filename = 'test_line_multiple_range.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        raises(ValueError, lambda: plot(x, y))
        p = plot(Piecewise((1, x > 0), (0, True)), (x, -1, 1))
        filename = 'test_plot_piecewise.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p = plot(Piecewise((x, x < 1), (x ** 2, True)), (x, -3, 3))
        filename = 'test_plot_piecewise_2.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p1 = plot(x)
        p2 = plot(3)
        p1.extend(p2)
        filename = 'test_horizontal_line.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        f = Piecewise((-1, x < -1), (x, And(-1 <= x, x < 0)), (x ** 2, And(0 <= x, x < 1)), (x ** 3, x >= 1))
        p = plot(f, (x, -3, 3))
        filename = 'test_plot_piecewise_3.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()