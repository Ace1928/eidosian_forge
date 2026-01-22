from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def _back_transform_out(self, xout, yout, params):
    try:
        yout[0][0, 0]
    except:
        pass
    else:
        return zip(*[self._back_transform_out(_x, _y, _p) for _x, _y, _p in zip(xout, yout, params)])
    x = xout if self.b_indep is None else self.b_indep(xout, yout, params).squeeze(axis=-1)
    y = yout if self.b_dep is None else self.b_dep(xout, yout, params)
    return (x, y, params)