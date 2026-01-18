import numpy as np
import pandas as pd
import pytest
from unittest import TestCase, SkipTest
from hvplot.util import (
def test_process_crs_pyproj_crs():
    pyproj = pytest.importorskip('pyproj')
    ccrs = pytest.importorskip('cartopy.crs')
    crs = process_crs(pyproj.CRS.from_epsg(4326))
    assert isinstance(crs, ccrs.PlateCarree)