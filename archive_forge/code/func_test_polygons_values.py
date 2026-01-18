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
def test_polygons_values(self):
    from geopandas.plotting import _plot_polygon_collection
    fig, ax = plt.subplots()
    coll = _plot_polygon_collection(ax, self.polygons, self.values)
    fig.canvas.draw_idle()
    cmap = plt.get_cmap()
    exp_colors = cmap(np.arange(self.N) / (self.N - 1))
    _check_colors(self.N, coll.get_facecolor(), exp_colors)
    ax.cla()
    coll = _plot_polygon_collection(ax, self.polygons, self.values, cmap='RdBu')
    fig.canvas.draw_idle()
    cmap = plt.get_cmap('RdBu')
    exp_colors = cmap(np.arange(self.N) / (self.N - 1))
    _check_colors(self.N, coll.get_facecolor(), exp_colors)
    ax.cla()
    coll = _plot_polygon_collection(ax, self.polygons, self.values, vmin=3, vmax=5)
    fig.canvas.draw_idle()
    cmap = plt.get_cmap()
    exp_colors = [cmap(0)]
    _check_colors(self.N, coll.get_facecolor(), exp_colors * 3)
    ax.cla()
    coll = _plot_polygon_collection(ax, self.polygons, self.values, edgecolor='g')
    fig.canvas.draw_idle()
    cmap = plt.get_cmap()
    exp_colors = cmap(np.arange(self.N) / (self.N - 1))
    _check_colors(self.N, coll.get_facecolor(), exp_colors)
    _check_colors(self.N, coll.get_edgecolor(), ['g'] * self.N)
    ax.cla()