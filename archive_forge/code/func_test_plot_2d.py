from sympy.external.importtools import import_module
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import (cos, sin)
def test_plot_2d():
    from sympy.plotting.pygletplot import PygletPlot
    p = PygletPlot(x, [x, -5, 5, 4], visible=False)
    p.wait_for_calculations()