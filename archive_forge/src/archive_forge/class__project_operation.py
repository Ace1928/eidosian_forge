import logging
import sys
import param
import numpy as np
import pandas as pd
from cartopy import crs as ccrs
from holoviews.core.data import MultiInterface
from holoviews.core.util import cartesian_product, get_param_values
from holoviews.operation import Operation
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.collection import GeometryCollection
from ..data import GeoPandasInterface
from ..element import (Image, Shape, Polygons, Path, Points, Contours,
from ..util import (
class _project_operation(Operation):
    """
    Baseclass for projection operations, projecting elements from their
    source coordinate reference system to the supplied projection.
    """
    projection = param.ClassSelector(default=ccrs.GOOGLE_MERCATOR, class_=ccrs.Projection, instantiate=False, doc='\n        Projection the shape type is projected to.')
    supported_types = []

    def _process(self, element, key=None):
        return element.map(self._process_element, self.supported_types)