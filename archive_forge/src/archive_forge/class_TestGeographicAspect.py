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
class TestGeographicAspect:

    def setup_class(self):
        pth = get_path('naturalearth_lowres')
        df = read_file(pth)
        self.north = df.loc[df.continent == 'North America']
        self.north_proj = self.north.to_crs('ESRI:102008')
        bounds = self.north.total_bounds
        y_coord = np.mean([bounds[1], bounds[3]])
        self.exp = 1 / np.cos(y_coord * np.pi / 180)

    def test_auto(self):
        ax = self.north.geometry.plot()
        assert ax.get_aspect() == self.exp
        ax2 = self.north_proj.geometry.plot()
        assert ax2.get_aspect() in ['equal', 1.0]
        ax = self.north.plot()
        assert ax.get_aspect() == self.exp
        ax2 = self.north_proj.plot()
        assert ax2.get_aspect() in ['equal', 1.0]
        ax3 = self.north.plot('pop_est')
        assert ax3.get_aspect() == self.exp
        ax4 = self.north_proj.plot('pop_est')
        assert ax4.get_aspect() in ['equal', 1.0]

    def test_manual(self):
        ax = self.north.geometry.plot(aspect='equal')
        assert ax.get_aspect() in ['equal', 1.0]
        self.north.geometry.plot(ax=ax, aspect=None)
        assert ax.get_aspect() in ['equal', 1.0]
        ax2 = self.north.geometry.plot(aspect=0.5)
        assert ax2.get_aspect() == 0.5
        self.north.geometry.plot(ax=ax2, aspect=None)
        assert ax2.get_aspect() == 0.5
        ax3 = self.north_proj.geometry.plot(aspect=0.5)
        assert ax3.get_aspect() == 0.5
        self.north_proj.geometry.plot(ax=ax3, aspect=None)
        assert ax3.get_aspect() == 0.5
        ax = self.north.plot(aspect='equal')
        assert ax.get_aspect() in ['equal', 1.0]
        self.north.plot(ax=ax, aspect=None)
        assert ax.get_aspect() in ['equal', 1.0]
        ax2 = self.north.plot(aspect=0.5)
        assert ax2.get_aspect() == 0.5
        self.north.plot(ax=ax2, aspect=None)
        assert ax2.get_aspect() == 0.5
        ax3 = self.north_proj.plot(aspect=0.5)
        assert ax3.get_aspect() == 0.5
        self.north_proj.plot(ax=ax3, aspect=None)
        assert ax3.get_aspect() == 0.5
        ax = self.north.plot('pop_est', aspect='equal')
        assert ax.get_aspect() in ['equal', 1.0]
        self.north.plot('pop_est', ax=ax, aspect=None)
        assert ax.get_aspect() in ['equal', 1.0]
        ax2 = self.north.plot('pop_est', aspect=0.5)
        assert ax2.get_aspect() == 0.5
        self.north.plot('pop_est', ax=ax2, aspect=None)
        assert ax2.get_aspect() == 0.5
        ax3 = self.north_proj.plot('pop_est', aspect=0.5)
        assert ax3.get_aspect() == 0.5
        self.north_proj.plot('pop_est', ax=ax3, aspect=None)
        assert ax3.get_aspect() == 0.5