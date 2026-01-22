from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def _forward_transform_xy(self, x, y, p):
    x, y, p = map(np.asarray, (x, y, p))
    if y.ndim == 1:
        _x = x if self.f_indep is None else self.f_indep(x, y[..., None, :], p[..., None, :])[..., 0]
        _y = y if self.f_dep is None else self.f_dep(x[..., 0], y, p)
        return (_x, _y, p)
    elif y.ndim == 2:
        return zip(*[self._forward_transform_xy(_x, _y, _p) for _x, _y, _p in zip(x, y, p)])
    else:
        raise NotImplementedError("Don't know what to do with %d dimensions." % y.ndim)