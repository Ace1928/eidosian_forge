import contextlib
import functools
import operator
import warnings
import numpy as np
from numpy.core import overrides
def _histogramdd_dispatcher(sample, bins=None, range=None, density=None, weights=None):
    if hasattr(sample, 'shape'):
        yield sample
    else:
        yield from sample
    with contextlib.suppress(TypeError):
        yield from bins
    yield weights