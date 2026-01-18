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
def test_single_color(self):
    ax = self.polys.plot(color='green')
    _check_colors(2, ax.collections[0].get_facecolors(), ['green'] * 2)
    assert len(ax.collections[0].get_edgecolors()) == 0
    ax = self.df.plot(color='green')
    _check_colors(2, ax.collections[0].get_facecolors(), ['green'] * 2)
    assert len(ax.collections[0].get_edgecolors()) == 0
    ax = self.df.plot(color=(0.5, 0.5, 0.5))
    _check_colors(2, ax.collections[0].get_facecolors(), [(0.5, 0.5, 0.5)] * 2)
    ax = self.df.plot(color=(0.5, 0.5, 0.5, 0.5))
    _check_colors(2, ax.collections[0].get_facecolors(), [(0.5, 0.5, 0.5, 0.5)] * 2)
    with pytest.raises((TypeError, ValueError)):
        self.df.plot(color='not color')
    with warnings.catch_warnings(record=True) as _:
        ax = self.df.plot(column='values', color='green')
        _check_colors(2, ax.collections[0].get_facecolors(), ['green'] * 2)