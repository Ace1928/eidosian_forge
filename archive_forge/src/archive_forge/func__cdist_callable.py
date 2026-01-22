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
def _cdist_callable(XA, XB, *, out, metric, **kwargs):
    mA = XA.shape[0]
    mB = XB.shape[0]
    dm = _prepare_out_argument(out, np.float64, (mA, mB))
    for i in range(mA):
        for j in range(mB):
            dm[i, j] = metric(XA[i], XB[j], **kwargs)
    return dm