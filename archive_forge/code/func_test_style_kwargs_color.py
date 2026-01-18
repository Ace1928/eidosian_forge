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
def test_style_kwargs_color(self):
    ax = self.polys.plot(facecolor='k')
    _check_colors(2, ax.collections[0].get_facecolors(), ['k'] * 2)
    ax = self.polys.plot(color='red', facecolor='k')
    ax = self.polys.plot(edgecolor='red')
    np.testing.assert_array_equal([(1, 0, 0, 1)], ax.collections[0].get_edgecolors())
    ax = self.df.plot('values', edgecolor='red')
    np.testing.assert_array_equal([(1, 0, 0, 1)], ax.collections[0].get_edgecolors())
    ax = self.polys.plot(facecolor='g', edgecolor='r', alpha=0.4)
    _check_colors(2, ax.collections[0].get_facecolors(), ['g'] * 2, alpha=0.4)
    _check_colors(2, ax.collections[0].get_edgecolors(), ['r'] * 2, alpha=0.4)
    ax = self.df.plot(facecolor=(0.5, 0.5, 0.5), edgecolor=(0.4, 0.5, 0.6))
    _check_colors(2, ax.collections[0].get_facecolors(), [(0.5, 0.5, 0.5)] * 2)
    _check_colors(2, ax.collections[0].get_edgecolors(), [(0.4, 0.5, 0.6)] * 2)
    ax = self.df.plot(facecolor=(0.5, 0.5, 0.5, 0.5), edgecolor=(0.4, 0.5, 0.6, 0.5))
    _check_colors(2, ax.collections[0].get_facecolors(), [(0.5, 0.5, 0.5, 0.5)] * 2)
    _check_colors(2, ax.collections[0].get_edgecolors(), [(0.4, 0.5, 0.6, 0.5)] * 2)