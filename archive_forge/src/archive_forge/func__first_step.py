from __future__ import absolute_import, division, print_function
import copy
import os
import warnings
from collections import defaultdict
import numpy as np
from .plotting import plot_result, plot_phase_plane
from .results import Result
from .util import _ensure_4args, _default
def _first_step(x, y):
    if len(_p) > 0:
        return self.first_step_cb(x, y, _p)
    else:
        return self.first_step_cb(x, y)