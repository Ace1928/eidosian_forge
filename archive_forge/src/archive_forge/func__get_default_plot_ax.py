import warnings
import numpy as np
from .chemistry import equilibrium_quotient, Equilibrium, Species
from .reactionsystem import ReactionSystem
from ._util import get_backend
from .util.pyutil import deprecated
from ._eqsys import EqCalcResult, NumSysLin, NumSysLog, NumSysSquare as _NumSysSquare
@staticmethod
def _get_default_plot_ax(subplot_kwargs=None):
    import matplotlib.pyplot as plt
    if subplot_kwargs is None:
        subplot_kwargs = dict(xscale='log', yscale='log')
    return plt.subplot(1, 1, 1, **subplot_kwargs)