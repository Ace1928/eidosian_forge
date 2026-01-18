from sympy.external.importtools import import_module
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import (cos, sin)
def test_plot_2d_discontinuous():
    from sympy.plotting.pygletplot import PygletPlot
    p = PygletPlot(1 / x, [x, -1, 1, 2], visible=False)
    p.wait_for_calculations()