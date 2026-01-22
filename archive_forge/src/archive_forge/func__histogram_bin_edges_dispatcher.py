import contextlib
import functools
import operator
import warnings
import numpy as np
from numpy.core import overrides
def _histogram_bin_edges_dispatcher(a, bins=None, range=None, weights=None):
    return (a, bins, weights)