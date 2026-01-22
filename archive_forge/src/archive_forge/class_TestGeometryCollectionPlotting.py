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
class TestGeometryCollectionPlotting:

    def setup_method(self):
        coll1 = GeometryCollection([Polygon([(1, 0), (2, 0), (2, 1)]), MultiLineString([((0.5, 0.5), (1, 1)), ((1, 0.5), (1.5, 1))])])
        coll2 = GeometryCollection([Point(0.75, 0.25), Polygon([(2, 2), (3, 2), (2, 3)])])
        self.series = GeoSeries([coll1, coll2])
        self.df = GeoDataFrame({'geometry': self.series, 'values': [1, 2]})

    def test_colors(self):
        ax = self.series.plot()
        _check_colors(2, ax.collections[0].get_facecolors(), [MPL_DFT_COLOR] * 2)
        _check_colors(2, ax.collections[1].get_edgecolors(), [MPL_DFT_COLOR] * 2)
        _check_colors(1, ax.collections[2].get_facecolors(), [MPL_DFT_COLOR])

    def test_values(self):
        ax = self.df.plot('values')
        cmap = plt.get_cmap()
        exp_colors = cmap([0.0, 1.0])
        _check_colors(2, ax.collections[0].get_facecolors(), exp_colors)
        _check_colors(2, ax.collections[1].get_edgecolors(), [exp_colors[0]] * 2)
        _check_colors(1, ax.collections[2].get_facecolors(), [exp_colors[1]])