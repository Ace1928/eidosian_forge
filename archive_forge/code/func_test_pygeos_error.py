from math import sqrt
from shapely.geometry import (
from numpy.testing import assert_array_equal
import geopandas
from geopandas import _compat as compat
from geopandas import GeoDataFrame, GeoSeries, read_file, datasets
import pytest
import numpy as np
import pandas as pd
@pytest.mark.skipif(compat.USE_SHAPELY_20 or not (compat.USE_PYGEOS and (not compat.PYGEOS_GE_010)), reason='PyGEOS < 0.10 does not support sindex.nearest')
def test_pygeos_error(self):
    df = geopandas.GeoDataFrame({'geometry': []})
    with pytest.raises(NotImplementedError, match='requires pygeos >= 0.10'):
        df.sindex.nearest(None)