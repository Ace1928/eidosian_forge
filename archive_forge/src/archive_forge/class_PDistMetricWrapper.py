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
class PDistMetricWrapper:
    metric_name: str

    def __call__(self, X, *, out=None, **kwargs):
        X = np.ascontiguousarray(X)
        m, n = X.shape
        metric_name = self.metric_name
        metric_info = _METRICS[metric_name]
        X, typ, kwargs = _validate_pdist_input(X, m, n, metric_info, **kwargs)
        out_size = m * (m - 1) // 2
        w = kwargs.pop('w', None)
        if w is not None:
            metric = metric_info.dist_func
            return _pdist_callable(X, metric=metric, out=out, w=w, **kwargs)
        dm = _prepare_out_argument(out, np.float64, (out_size,))
        pdist_fn = getattr(_distance_wrap, f'pdist_{metric_name}_{typ}_wrap')
        pdist_fn(X, dm, **kwargs)
        return dm