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
class collapse(Operation):
    """
    Given an overlay of Element types, collapse into single Element
    object using supplied function. Collapsing aggregates over the
    key dimensions of each object applying the supplied fn to each group.

    This is an example of an Operation that does not involve
    any Raster types.
    """
    fn = param.Callable(default=np.mean, doc='\n        The function that is used to collapse the curve y-values for\n        each x-value.')

    def _process(self, overlay, key=None):
        if isinstance(overlay, NdOverlay):
            collapse_map = HoloMap(overlay)
        else:
            collapse_map = HoloMap({i: el for i, el in enumerate(overlay)})
        return collapse_map.collapse(function=self.p.fn)