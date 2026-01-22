from __future__ import absolute_import, division, print_function
import inspect
import math
import os
import warnings
import sys
import numpy as np
from .plotting import plot_series
def _mpf(val):
    try:
        return mp.mpf(val)
    except TypeError:
        return mp.mpf(float(val))