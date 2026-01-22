import math
from functools import partial
import numpy as np
import param
from ..core import NdOverlay, Overlay
from ..element.chart import Area
from .resample import ResampleOperation1D
class downsample1d(ResampleOperation1D):
    """
    Implements downsampling of a regularly sampled 1D dataset.

    Supports multiple algorithms:

        - `lttb`: Largest Triangle Three Buckets downsample algorithm
        - `nth`: Selects every n-th point.
    """
    algorithm = param.Selector(default='lttb', objects=['lttb', 'nth'])

    def _process(self, element, key=None):
        if isinstance(element, (Overlay, NdOverlay)):
            _process = partial(self._process, key=key)
            if isinstance(element, Overlay):
                elements = [v.map(_process) for v in element]
            else:
                elements = {k: v.map(_process) for k, v in element.items()}
            return element.clone(elements)
        if self.p.x_range:
            element = element[slice(*self.p.x_range)]
        if len(element) <= self.p.width:
            return element
        xs, ys = (element.dimension_values(i) for i in range(2))
        if ys.dtype == np.bool_:
            ys = ys.astype(np.int8)
        downsample = _ALGORITHMS[self.p.algorithm]
        if self.p.algorithm == 'lttb' and isinstance(element, Area):
            raise NotImplementedError('LTTB algorithm is not implemented for hv.Area')
        samples = downsample(xs, ys, self.p.width)
        return element.iloc[samples]