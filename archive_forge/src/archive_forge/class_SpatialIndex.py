import warnings
from shapely.geometry.base import BaseGeometry
import pandas as pd
import numpy as np
from . import _compat as compat
from ._decorator import doc
class SpatialIndex(rtree.index.Index, BaseSpatialIndex):
    """Original rtree wrapper, kept for backwards compatibility."""

    def __init__(self, *args):
        warnings.warn('Directly using SpatialIndex is deprecated, and the class will be removed in a future version. Access the spatial index through the `GeoSeries.sindex` attribute, or use `rtree.index.Index` directly.', FutureWarning, stacklevel=2)
        super().__init__(*args)

    @doc(BaseSpatialIndex.intersection)
    def intersection(self, coordinates, *args, **kwargs):
        return super().intersection(coordinates, *args, **kwargs)

    @doc(BaseSpatialIndex.nearest)
    def nearest(self, *args, **kwargs):
        return super().nearest(*args, **kwargs)

    @property
    @doc(BaseSpatialIndex.size)
    def size(self):
        return len(self.leaves()[0][1])

    @property
    @doc(BaseSpatialIndex.is_empty)
    def is_empty(self):
        if len(self.leaves()) > 1:
            return False
        return self.size < 1