import numpy as np
import pandas as pd
import pytest
from unittest import TestCase, SkipTest
from hvplot.util import (
def test_process_crs_raises_error():
    pytest.importorskip('pyproj')
    pytest.importorskip('cartopy.crs')
    with pytest.raises(ValueError, match='must be defined as a EPSG code, proj4 string'):
        process_crs(43823)