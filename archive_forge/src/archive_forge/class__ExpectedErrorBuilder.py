import os
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal
import pytest
from .test_file import FIONA_MARK, PYOGRIO_MARK
class _ExpectedErrorBuilder:

    def __init__(self, composite_key):
        self.composite_key = composite_key

    def to_raise(self, error_type, error_match):
        _expected_exceptions[self.composite_key] = _ExpectedError(error_type, error_match)