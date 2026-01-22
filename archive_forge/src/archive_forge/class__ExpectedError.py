import os
from shapely.geometry import (
import geopandas
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal
import pytest
from .test_file import FIONA_MARK, PYOGRIO_MARK
class _ExpectedError:

    def __init__(self, error_type, error_message_match):
        self.type = error_type
        self.match = error_message_match