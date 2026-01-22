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
@dataclasses.dataclass(frozen=True)
class CDistWeightedMetricWrapper:
    metric_name: str
    weighted_metric: str

    def __call__(self, XA, XB, *, out=None, **kwargs):
        XA = np.ascontiguousarray(XA)
        XB = np.ascontiguousarray(XB)
        mA, n = XA.shape
        mB, _ = XB.shape
        metric_name = self.metric_name
        XA, XB, typ, kwargs = _validate_cdist_input(XA, XB, mA, mB, n, _METRICS[metric_name], **kwargs)
        dm = _prepare_out_argument(out, np.float64, (mA, mB))
        w = kwargs.pop('w', None)
        if w is not None:
            metric_name = self.weighted_metric
            kwargs['w'] = w
        cdist_fn = getattr(_distance_wrap, f'cdist_{metric_name}_{typ}_wrap')
        cdist_fn(XA, XB, dm, **kwargs)
        return dm