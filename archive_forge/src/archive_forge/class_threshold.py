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
class threshold(Operation):
    """
    Threshold a given Image whereby all values higher than a given
    level map to the specified high value and all values lower than
    that level map to the specified low value.
    """
    output_type = Image
    level = param.Number(default=0.5, doc="\n       The value at which the threshold is applied. Values lower than\n       the threshold map to the 'low' value and values above map to\n       the 'high' value.")
    high = param.Number(default=1.0, doc='\n      The value given to elements greater than (or equal to) the\n      threshold.')
    low = param.Number(default=0.0, doc='\n      The value given to elements below the threshold.')
    group = param.String(default='Threshold', doc='\n       The group assigned to the thresholded output.')
    _per_element = True

    def _process(self, matrix, key=None):
        if not isinstance(matrix, Image):
            raise TypeError('The threshold operation requires a Image as input.')
        arr = matrix.data
        high = np.ones(arr.shape) * self.p.high
        low = np.ones(arr.shape) * self.p.low
        thresholded = np.where(arr > self.p.level, high, low)
        return matrix.clone(thresholded, group=self.p.group)