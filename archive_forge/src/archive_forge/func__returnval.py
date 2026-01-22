import os
import atexit
import functools
import pickle
import sys
import time
import warnings
import numpy as np
def _returnval(self, a, b):
    """Behave correctly when working on scalars/arrays.

        Either input is an array and we in-place write b (output from
        mpi4py) back into a, or input is a scalar and we return the
        corresponding output scalar."""
    if np.isscalar(a):
        assert np.isscalar(b)
        return b
    else:
        assert not np.isscalar(b)
        a[:] = b
        return None