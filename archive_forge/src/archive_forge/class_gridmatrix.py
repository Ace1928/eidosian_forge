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
class gridmatrix(param.ParameterizedFunction):
    """
    The gridmatrix operation takes an Element or HoloMap
    of Elements as input and creates a GridMatrix object,
    which plots each dimension in the Element against
    each other dimension. This provides a very useful
    overview of high-dimensional data and is inspired
    by pandas and seaborn scatter_matrix implementations.
    """
    chart_type = param.Parameter(default=Scatter, doc='\n        The Element type used to display bivariate distributions\n        of the data.')
    diagonal_type = param.Parameter(default=None, doc='\n       The Element type along the diagonal, may be a Histogram or any\n       other plot type which can visualize a univariate distribution.\n       This parameter overrides diagonal_operation.')
    diagonal_operation = param.Parameter(default=histogram, doc='\n       The operation applied along the diagonal, may be a histogram-operation\n       or any other function which returns a viewable element.')
    overlay_dims = param.List(default=[], doc='\n       If a HoloMap is supplied, this will allow overlaying one or\n       more of its key dimensions.')

    def __call__(self, data, **params):
        p = param.ParamOverrides(self, params)
        if isinstance(data, (HoloMap, NdOverlay)):
            ranges = {d.name: data.range(d) for d in data.dimensions()}
            data = data.clone({k: GridMatrix(self._process(p, v, ranges)) for k, v in data.items()})
            data = Collator(data, merge_type=type(data))()
            if p.overlay_dims:
                data = data.map(lambda x: x.overlay(p.overlay_dims), (HoloMap,))
            return data
        elif isinstance(data, Element):
            data = self._process(p, data)
            return GridMatrix(data)

    def _process(self, p, element, ranges=None):
        if ranges is None:
            ranges = {}
        if isinstance(element.data, np.ndarray):
            el_data = element.table(default_datatype)
        else:
            el_data = element.data
        types = (str, np.str_, np.object_) + datetime_types
        dims = [d for d in element.dimensions() if _is_number(element.range(d)[0]) and (not issubclass(element.get_dimension_type(d), types))]
        permuted_dims = [(d1, d2) for d1 in dims for d2 in dims[::-1]]
        if p.diagonal_type is Histogram:
            p.diagonal_type = None
            p.diagonal_operation = histogram
        data = {}
        for d1, d2 in permuted_dims:
            if d1 == d2:
                if p.diagonal_type is not None:
                    if p.diagonal_type._auto_indexable_1d:
                        el = p.diagonal_type(el_data, kdims=[d1], vdims=[d2], datatype=[default_datatype])
                    else:
                        values = element.dimension_values(d1)
                        el = p.diagonal_type(values, kdims=[d1])
                elif p.diagonal_operation is None:
                    continue
                elif p.diagonal_operation is histogram or isinstance(p.diagonal_operation, histogram):
                    bin_range = ranges.get(d1.name, element.range(d1))
                    el = p.diagonal_operation(element, dimension=d1.name, bin_range=bin_range)
                else:
                    el = p.diagonal_operation(element, dimension=d1.name)
            else:
                kdims, vdims = ([d1, d2], []) if len(p.chart_type.kdims) == 2 else (d1, d2)
                el = p.chart_type(el_data, kdims=kdims, vdims=vdims, datatype=[default_datatype])
            data[d1.name, d2.name] = el
        return data