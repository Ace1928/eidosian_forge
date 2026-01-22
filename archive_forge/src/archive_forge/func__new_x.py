from __future__ import absolute_import, division, print_function
import copy
import os
import warnings
from collections import defaultdict
import numpy as np
from .plotting import plot_result, plot_phase_plane
from .results import Result
from .util import _ensure_4args, _default
def _new_x(xout, x, guaranteed_autonomous):
    if guaranteed_autonomous:
        return (0, abs(x[-1] - xout[-1]))
    else:
        return (xout[-1], x[-1])