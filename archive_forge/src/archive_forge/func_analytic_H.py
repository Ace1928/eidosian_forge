from collections import defaultdict
from functools import reduce
from operator import mul
import pytest
from chempy import ReactionSystem
from chempy.units import (
from chempy.util.testing import requires
from ..integrated import binary_rev
from ..ode import get_odesys
from .._native import get_native
def analytic_H(t, p, k, H0):
    x0 = np.sqrt(2) * np.sqrt(p)
    x1 = x0
    x2 = np.sqrt(k)
    x3 = t * x1 * x2
    x4 = H0 * x2
    x5 = np.sqrt(x0 + 2 * x4)
    x6 = np.sqrt(-1 / (2 * H0 * x2 - x0))
    x7 = x5 * x6 * np.exp(x3)
    x8 = np.exp(-x3) / (x5 * x6)
    return x1 * (x7 - x8) / (2 * x2 * (x7 + x8))