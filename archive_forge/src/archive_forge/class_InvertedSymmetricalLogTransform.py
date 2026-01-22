import inspect
import textwrap
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.ticker import (
from matplotlib.transforms import Transform, IdentityTransform
class InvertedSymmetricalLogTransform(Transform):
    input_dims = output_dims = 1

    def __init__(self, base, linthresh, linscale):
        super().__init__()
        symlog = SymmetricalLogTransform(base, linthresh, linscale)
        self.base = base
        self.linthresh = linthresh
        self.invlinthresh = symlog.transform(linthresh)
        self.linscale = linscale
        self._linscale_adj = linscale / (1.0 - self.base ** (-1))

    @_api.rename_parameter('3.8', 'a', 'values')
    def transform_non_affine(self, values):
        abs_a = np.abs(values)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.sign(values) * self.linthresh * np.power(self.base, abs_a / self.linthresh - self._linscale_adj)
            inside = abs_a <= self.invlinthresh
        out[inside] = values[inside] / self._linscale_adj
        return out

    def inverted(self):
        return SymmetricalLogTransform(self.base, self.linthresh, self.linscale)