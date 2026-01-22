import colorsys
from copy import deepcopy
from operator import itemgetter
import numpy as np
import param
from ..core import Dataset, Dimension, Element2D, Overlay, config, util
from ..core.boundingregion import BoundingBox, BoundingRegion
from ..core.data import ImageInterface
from ..core.data.interface import DataError
from ..core.dimension import dimension_name
from ..core.sheetcoords import SheetCoordinateSystem, Slice
from .chart import Curve
from .geom import Selection2DExpr
from .graphs import TriMesh
from .tabular import Table
from .util import categorical_aggregate2d, compute_slice_bounds
class HeatMap(Selection2DExpr, Dataset, Element2D):
    """
    HeatMap represents a 2D grid of categorical coordinates which can
    be computed from a sparse tabular representation. A HeatMap does
    not automatically aggregate the supplied values, so if the data
    contains multiple entries for the same coordinate on the 2D grid
    it should be aggregated using the aggregate method before display.

    The HeatMap constructor will support any tabular or gridded data
    format with 2 coordinates and at least one value dimension. A
    simple example:

        HeatMap([(x1, y1, z1), (x2, y2, z2), ...])

    However any tabular and gridded format, including pandas
    DataFrames, dictionaries of columns, xarray DataArrays and more
    are supported if the library is importable.
    """
    group = param.String(default='HeatMap', constant=True)
    kdims = param.List(default=[Dimension('x'), Dimension('y')], bounds=(2, 2), constant=True)
    vdims = param.List(default=[Dimension('z')], constant=True)

    def __init__(self, data, kdims=None, vdims=None, **params):
        super().__init__(data, kdims=kdims, vdims=vdims, **params)
        self._gridded = None

    @property
    def gridded(self):
        if self._gridded is None:
            self._gridded = categorical_aggregate2d(self)
        return self._gridded

    @property
    def _unique(self):
        """
        Reports if the Dataset is unique.
        """
        return self.gridded.label != 'non-unique'

    def range(self, dim, data_range=True, dimension_range=True):
        """Return the lower and upper bounds of values along dimension.

        Args:
            dimension: The dimension to compute the range on.
            data_range (bool): Compute range from data values
            dimension_range (bool): Include Dimension ranges
                Whether to include Dimension range and soft_range
                in range calculation

        Returns:
            Tuple containing the lower and upper bound
        """
        dim = self.get_dimension(dim)
        if dim in self.kdims:
            try:
                self.gridded._binned = True
                if self.gridded is self:
                    return super().range(dim, data_range, dimension_range)
                else:
                    drange = self.gridded.range(dim, data_range, dimension_range)
            except Exception:
                drange = None
            finally:
                self.gridded._binned = False
            if drange is not None:
                return drange
        return super().range(dim, data_range, dimension_range)