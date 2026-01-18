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
def test_plot_and_save_2():
    if not matplotlib:
        skip('Matplotlib not the default backend')
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    with TemporaryDirectory(prefix='sympy_') as tmpdir:
        p = plot_parametric(sin(x), cos(x))
        filename = 'test_parametric.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p = plot_parametric(sin(x), cos(x), (x, -5, 5), legend=True, label='parametric_plot')
        filename = 'test_parametric_range.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p = plot_parametric((sin(x), cos(x)), (x, sin(x)))
        filename = 'test_parametric_multiple.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p = plot_parametric((sin(x), cos(x), (x, -3, 3)), (x, sin(x), (x, -5, 5)))
        filename = 'test_parametric_multiple_ranges.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p = plot_parametric(x, sin(x), depth=13)
        filename = 'test_recursion_depth.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p = plot_parametric(cos(x), sin(x), adaptive=False, nb_of_points=500)
        filename = 'test_adaptive.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p = plot3d_parametric_line(sin(x), cos(x), x, legend=True, label='3d_parametric_plot')
        filename = 'test_3d_line.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p = plot3d_parametric_line((sin(x), cos(x), x, (x, -5, 5)), (cos(x), sin(x), x, (x, -3, 3)))
        filename = 'test_3d_line_multiple.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p = plot3d_parametric_line(sin(x), cos(x), x, nb_of_points=30)
        filename = 'test_3d_line_points.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p = plot3d(x * y)
        filename = 'test_surface.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p = plot3d(-x * y, x * y, (x, -5, 5))
        filename = 'test_surface_multiple.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p = plot3d((x * y, (x, -3, 3), (y, -3, 3)), (-x * y, (x, -3, 3), (y, -3, 3)))
        filename = 'test_surface_multiple_ranges.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p = plot3d_parametric_surface(sin(x + y), cos(x - y), x - y)
        filename = 'test_parametric_surface.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p = plot3d_parametric_surface((x * sin(z), x * cos(z), z, (x, -5, 5), (z, -5, 5)), (sin(x + y), cos(x - y), x - y, (x, -5, 5), (y, -5, 5)))
        filename = 'test_parametric_surface.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p = plot_contour(sin(x) * sin(y), (x, -5, 5), (y, -5, 5))
        filename = 'test_contour_plot.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p = plot_contour(x ** 2 + y ** 2, x ** 3 + y ** 3, (x, -5, 5), (y, -5, 5))
        filename = 'test_contour_plot.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()
        p = plot_contour((x ** 2 + y ** 2, (x, -5, 5), (y, -5, 5)), (x ** 3 + y ** 3, (x, -3, 3), (y, -3, 3)))
        filename = 'test_contour_plot.png'
        p.save(os.path.join(tmpdir, filename))
        p._backend.close()