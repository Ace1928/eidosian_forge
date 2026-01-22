import pyglet.gl as pgl
from sympy.core import S
from sympy.plotting.pygletplot.color_scheme import ColorScheme
from sympy.plotting.pygletplot.plot_mode import PlotMode
from sympy.utilities.iterables import is_sequence
from time import sleep
from threading import Thread, Event, RLock
import warnings
def _get_evaluator(self):
    if self.use_lambda_eval:
        try:
            e = self._get_lambda_evaluator()
            return e
        except Exception:
            warnings.warn('\nWarning: creating lambda evaluator failed. Falling back on SymPy subs evaluator.')
    return self._get_sympy_evaluator()