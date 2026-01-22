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
class TestGeoplotAccessor:

    def setup_method(self):
        geometries = [Polygon([(0, 0), (1, 0), (1, 1)]), Point(1, 3)]
        x = [1, 2]
        y = [10, 20]
        self.gdf = GeoDataFrame({'geometry': geometries, 'x': x, 'y': y}, crs='EPSG:4326')
        self.df = pd.DataFrame({'x': x, 'y': y})

    def compare_figures(self, kind, fig_test, fig_ref, kwargs):
        """Compare Figures."""
        ax_pandas_1 = fig_test.subplots()
        self.df.plot(kind=kind, ax=ax_pandas_1, **kwargs)
        ax_geopandas_1 = fig_ref.subplots()
        self.gdf.plot(kind=kind, ax=ax_geopandas_1, **kwargs)
        ax_pandas_2 = fig_test.subplots()
        getattr(self.df.plot, kind)(ax=ax_pandas_2, **kwargs)
        ax_geopandas_2 = fig_ref.subplots()
        getattr(self.gdf.plot, kind)(ax=ax_geopandas_2, **kwargs)
    _pandas_kinds = GeoplotAccessor._pandas_kinds
    if MPL_DECORATORS:

        @pytest.mark.parametrize('kind', _pandas_kinds)
        @check_figures_equal(extensions=['png', 'pdf'])
        def test_pandas_kind(self, kind, fig_test, fig_ref):
            """Test Pandas kind."""
            import importlib
            _scipy_dependent_kinds = ['kde', 'density']
            _y_kinds = ['pie']
            _xy_kinds = ['scatter', 'hexbin']
            kwargs = {}
            if kind in _scipy_dependent_kinds:
                if not importlib.util.find_spec('scipy'):
                    with pytest.raises(ModuleNotFoundError, match="No module named 'scipy'"):
                        self.gdf.plot(kind=kind)
                    return
            elif kind in _y_kinds:
                kwargs = {'y': 'y'}
            elif kind in _xy_kinds:
                kwargs = {'x': 'x', 'y': 'y'}
                if kind == 'hexbin':
                    kwargs['gridsize'] = 10
            self.compare_figures(kind, fig_test, fig_ref, kwargs)
            plt.close('all')

        @check_figures_equal(extensions=['png', 'pdf'])
        def test_geo_kind(self, fig_test, fig_ref):
            """Test Geo kind."""
            ax1 = fig_test.subplots()
            self.gdf.plot(ax=ax1)
            ax2 = fig_ref.subplots()
            getattr(self.gdf.plot, 'geo')(ax=ax2)
            plt.close('all')

    def test_invalid_kind(self):
        """Test invalid kinds."""
        with pytest.raises(ValueError, match='error is not a valid plot kind'):
            self.gdf.plot(kind='error')
        with pytest.raises(AttributeError, match="'GeoplotAccessor' object has no attribute 'error'"):
            self.gdf.plot.error()