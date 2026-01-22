from sympy.external.importtools import import_module
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import (cos, sin)
def _test_plot_log():
    from sympy.plotting.pygletplot import PygletPlot
    p = PygletPlot(log(x), [x, 0, 6.282, 4], 'mode=polar', visible=False)
    p.wait_for_calculations()