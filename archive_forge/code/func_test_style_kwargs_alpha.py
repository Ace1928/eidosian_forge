import itertools
from packaging.version import Version
import warnings
import numpy as np
import pandas as pd
from shapely import wkt
from shapely.affinity import rotate
from shapely.geometry import (
from geopandas import GeoDataFrame, GeoSeries, read_file
from geopandas.datasets import get_path
import geopandas._compat as compat
from geopandas.plotting import GeoplotAccessor
import pytest
import matplotlib.pyplot as plt
def test_style_kwargs_alpha(self):
    ax = self.df.plot(alpha=0.7)
    np.testing.assert_array_equal([0.7], ax.collections[0].get_alpha())