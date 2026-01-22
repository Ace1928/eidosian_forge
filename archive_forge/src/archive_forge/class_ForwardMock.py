import pandas as pd
from shapely.geometry import Point
from geopandas import GeoDataFrame, GeoSeries
from geopandas.tools import geocode, reverse_geocode
from geopandas.tools.geocoding import _prepare_geocode_result
from geopandas.tests.util import assert_geoseries_equal, mock
from pandas.testing import assert_series_equal
from geopandas.testing import assert_geodataframe_equal
import pytest
class ForwardMock(mock.MagicMock):
    """
    Mock the forward geocoding function.
    Returns the passed in address and (p, p+.5) where p increases
    at each call

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n = 0.0

    def __call__(self, *args, **kwargs):
        self.return_value = (args[0], (self._n, self._n + 0.5))
        self._n += 1
        return super().__call__(*args, **kwargs)