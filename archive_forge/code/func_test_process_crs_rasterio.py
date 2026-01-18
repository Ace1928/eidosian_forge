import numpy as np
import pandas as pd
import pytest
from unittest import TestCase, SkipTest
from hvplot.util import (
def test_process_crs_rasterio():
    pytest.importorskip('pyproj')
    rcrs = pytest.importorskip('rasterio.crs')
    ccrs = pytest.importorskip('cartopy.crs')
    input = rcrs.CRS.from_epsg(4326).to_wkt()
    crs = process_crs(input)
    assert isinstance(crs, ccrs.CRS)