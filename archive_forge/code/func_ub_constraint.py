import functools
from threading import RLock
import numpy as np
from scipy.optimize import _cobyla as cobyla
from ._optimize import (OptimizeResult, _check_unknown_options,
def ub_constraint(x):
    return bounds.ub[i_ub] - x[i_ub]