from sympy.external.importtools import import_module
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import (cos, sin)
def test_plot_integral():
    from sympy.plotting.pygletplot import PygletPlot
    from sympy.integrals.integrals import Integral
    p = PygletPlot(Integral(z * x, (x, 1, z), (z, 1, y)), visible=False)
    p.wait_for_calculations()