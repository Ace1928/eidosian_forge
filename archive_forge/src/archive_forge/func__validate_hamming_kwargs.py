import math
import warnings
import numpy as np
import dataclasses
from typing import Optional, Callable
from functools import partial
from scipy._lib._util import _asarray_validated
from . import _distance_wrap
from . import _hausdorff
from ..linalg import norm
from ..special import rel_entr
from . import _distance_pybind
def _validate_hamming_kwargs(X, m, n, **kwargs):
    w = kwargs.get('w', np.ones((n,), dtype='double'))
    if w.ndim != 1 or w.shape[0] != n:
        raise ValueError('Weights must have same size as input vector. %d vs. %d' % (w.shape[0], n))
    kwargs['w'] = _validate_weights(w)
    return kwargs