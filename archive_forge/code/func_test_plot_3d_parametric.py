from sympy.external.importtools import import_module
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import (cos, sin)
def test_plot_3d_parametric():
    from sympy.plotting.pygletplot import PygletPlot
    p = PygletPlot(sin(x), cos(x), x / 5.0, [x, 0, 6.282, 4], visible=False)
    p.wait_for_calculations()