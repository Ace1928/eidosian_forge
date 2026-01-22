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
class decimate(Operation):
    """
    Decimates any column based Element to a specified number of random
    rows if the current element defined by the x_range and y_range
    contains more than max_samples. By default the operation returns a
    DynamicMap with a RangeXY stream allowing dynamic downsampling.
    """
    dynamic = param.Boolean(default=True, doc='\n       Enables dynamic processing by default.')
    link_inputs = param.Boolean(default=True, doc='\n         By default, the link_inputs parameter is set to True so that\n         when applying shade, backends that support linked streams\n         update RangeXY streams on the inputs of the shade operation.')
    max_samples = param.Integer(default=5000, doc='\n        Maximum number of samples to display at the same time.')
    random_seed = param.Integer(default=42, doc='\n        Seed used to initialize randomization.')
    streams = param.ClassSelector(default=[RangeXY], class_=(dict, list), doc='\n        List of streams that are applied if dynamic=True, allowing\n        for dynamic interaction with the plot.')
    x_range = param.NumericTuple(default=None, length=2, doc='\n       The x_range as a tuple of min and max x-value. Auto-ranges\n       if set to None.')
    y_range = param.NumericTuple(default=None, length=2, doc='\n       The x_range as a tuple of min and max y-value. Auto-ranges\n       if set to None.')
    _per_element = True

    def _process_layer(self, element, key=None):
        if not isinstance(element, Dataset):
            raise ValueError('Cannot downsample non-Dataset types.')
        if element.interface not in column_interfaces:
            element = element.clone(tuple(element.columns().values()))
        xstart, xend = self.p.x_range if self.p.x_range else element.range(0)
        ystart, yend = self.p.y_range if self.p.y_range else element.range(1)
        xdim, ydim = element.dimensions(label=True)[0:2]
        sliced = element.select(**{xdim: (xstart, xend), ydim: (ystart, yend)})
        if len(sliced) > self.p.max_samples:
            prng = np.random.RandomState(self.p.random_seed)
            choice = prng.choice(len(sliced), self.p.max_samples, False)
            return sliced.iloc[np.sort(choice)]
        return sliced

    def _process(self, element, key=None):
        return element.map(self._process_layer, Element)