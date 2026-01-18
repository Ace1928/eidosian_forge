import warnings
from shapely.geometry.base import BaseGeometry
import pandas as pd
import numpy as np
from . import _compat as compat
from ._decorator import doc
@property
def valid_query_predicates(self):
    """Returns valid predicates for the used spatial index.

            Returns
            -------
            set
                Set of valid predicates for this spatial index.

            Examples
            --------
            >>> from shapely.geometry import Point
            >>> s = geopandas.GeoSeries([Point(0, 0), Point(1, 1)])
            >>> s.sindex.valid_query_predicates  # doctest: +SKIP
            {None, "contains", "contains_properly", "covered_by", "covers", "crosses", "intersects", "overlaps", "touches", "within"}
            """
    return _PYGEOS_PREDICATES