import pandas as pd
import pyproj
import pytest
from shapely.geometry import Point
import numpy as np
from geopandas import GeoDataFrame, GeoSeries
import geopandas

    Helper method to make tests easier to read. Checks result is of the expected
    type. Asserts that accessing result.geometry.name raises, corresponding to
    _geometry_column_name being in an invalid state
    (either None, or a column no longer present)
    This amounts to testing the assertion raised (geometry column is unset, vs
    old geometry column is missing)

    We assert that _geometry_column_name = int_geo_colname

    