import inspect
import textwrap
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.ticker import (
from matplotlib.transforms import Transform, IdentityTransform
class LogTransform(Transform):
    input_dims = output_dims = 1

    def __init__(self, base, nonpositive='clip'):
        super().__init__()
        if base <= 0 or base == 1:
            raise ValueError('The log base cannot be <= 0 or == 1')
        self.base = base
        self._clip = _api.check_getitem({'clip': True, 'mask': False}, nonpositive=nonpositive)

    def __str__(self):
        return '{}(base={}, nonpositive={!r})'.format(type(self).__name__, self.base, 'clip' if self._clip else 'mask')

    @_api.rename_parameter('3.8', 'a', 'values')
    def transform_non_affine(self, values):
        with np.errstate(divide='ignore', invalid='ignore'):
            log = {np.e: np.log, 2: np.log2, 10: np.log10}.get(self.base)
            if log:
                out = log(values)
            else:
                out = np.log(values)
                out /= np.log(self.base)
            if self._clip:
                out[values <= 0] = -1000
        return out

    def inverted(self):
        return InvertedLogTransform(self.base)