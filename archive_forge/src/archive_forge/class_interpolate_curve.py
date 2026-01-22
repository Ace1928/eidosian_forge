import warnings
import numpy as np
import param
from packaging.version import Version
from param import _is_number
from ..core import (
from ..core.data import ArrayInterface, DictInterface, PandasInterface, default_datatype
from ..core.data.util import dask_array_module
from ..core.util import (
from ..element.chart import Histogram, Scatter
from ..element.path import Contours, Polygons
from ..element.raster import RGB, Image
from ..element.util import categorical_aggregate2d  # noqa (API import)
from ..streams import RangeXY
from ..util.locator import MaxNLocator
class interpolate_curve(Operation):
    """
    Resamples a Curve using the defined interpolation method, e.g.
    to represent changes in y-values as steps.
    """
    interpolation = param.ObjectSelector(objects=['steps-pre', 'steps-mid', 'steps-post', 'linear'], default='steps-mid', doc='\n       Controls the transition point of the step along the x-axis.')
    _per_element = True

    @classmethod
    def pts_to_prestep(cls, x, values):
        steps = np.zeros(2 * len(x) - 1)
        value_steps = tuple((np.empty(2 * len(x) - 1, dtype=v.dtype) for v in values))
        steps[0::2] = x
        steps[1::2] = steps[0:-2:2]
        val_arrays = []
        for v, s in zip(values, value_steps):
            s[0::2] = v
            s[1::2] = s[2::2]
            val_arrays.append(s)
        return (steps, tuple(val_arrays))

    @classmethod
    def pts_to_midstep(cls, x, values):
        steps = np.zeros(2 * len(x))
        value_steps = tuple((np.empty(2 * len(x), dtype=v.dtype) for v in values))
        steps[1:-1:2] = steps[2::2] = x[:-1] + (x[1:] - x[:-1]) / 2
        steps[0], steps[-1] = (x[0], x[-1])
        val_arrays = []
        for v, s in zip(values, value_steps):
            s[0::2] = v
            s[1::2] = s[0::2]
            val_arrays.append(s)
        return (steps, tuple(val_arrays))

    @classmethod
    def pts_to_poststep(cls, x, values):
        steps = np.zeros(2 * len(x) - 1)
        value_steps = tuple((np.empty(2 * len(x) - 1, dtype=v.dtype) for v in values))
        steps[0::2] = x
        steps[1::2] = steps[2::2]
        val_arrays = []
        for v, s in zip(values, value_steps):
            s[0::2] = v
            s[1::2] = s[0:-2:2]
            val_arrays.append(s)
        return (steps, tuple(val_arrays))

    def _process_layer(self, element, key=None):
        INTERPOLATE_FUNCS = {'steps-pre': self.pts_to_prestep, 'steps-mid': self.pts_to_midstep, 'steps-post': self.pts_to_poststep}
        if self.p.interpolation not in INTERPOLATE_FUNCS:
            return element
        x = element.dimension_values(0)
        is_datetime = isdatetime(x)
        if is_datetime:
            dt_type = 'datetime64[ns]'
            x = x.astype(dt_type)
        dvals = tuple((element.dimension_values(d) for d in element.dimensions()[1:]))
        xs, dvals = INTERPOLATE_FUNCS[self.p.interpolation](x, dvals)
        if is_datetime:
            xs = xs.astype(dt_type)
        return element.clone((xs,) + dvals)

    def _process(self, element, key=None):
        return element.map(self._process_layer, Element)