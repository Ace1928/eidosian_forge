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
def test_style_kwargs_linestyle(self):
    for ax in [self.series.plot(linestyle=':', linewidth=1), self.df.plot(linestyle=':', linewidth=1)]:
        assert [(0.0, [1.0, 1.65])] == ax.collections[0].get_linestyle()
    ax = self.series.plot(linestyle=(0, (3, 10, 1, 15)), linewidth=1)
    assert [(0, [3, 10, 1, 15])] == ax.collections[0].get_linestyle()