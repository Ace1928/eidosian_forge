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
class TestPointPlotting:

    def setup_method(self):
        self.N = 10
        self.points = GeoSeries((Point(i, i) for i in range(self.N)))
        values = np.arange(self.N)
        self.df = GeoDataFrame({'geometry': self.points, 'values': values})
        self.df['exp'] = (values * 10) ** 3
        multipoint1 = MultiPoint(self.points)
        multipoint2 = rotate(multipoint1, 90)
        self.df2 = GeoDataFrame({'geometry': [multipoint1, multipoint2], 'values': [0, 1]})

    def test_figsize(self):
        ax = self.points.plot(figsize=(1, 1))
        np.testing.assert_array_equal(ax.figure.get_size_inches(), (1, 1))
        ax = self.df.plot(figsize=(1, 1))
        np.testing.assert_array_equal(ax.figure.get_size_inches(), (1, 1))

    def test_default_colors(self):
        ax = self.points.plot()
        _check_colors(self.N, ax.collections[0].get_facecolors(), [MPL_DFT_COLOR] * self.N)
        ax = self.df.plot()
        _check_colors(self.N, ax.collections[0].get_facecolors(), [MPL_DFT_COLOR] * self.N)
        ax = self.df.plot(column='values')
        cmap = plt.get_cmap()
        expected_colors = cmap(np.arange(self.N) / (self.N - 1))
        _check_colors(self.N, ax.collections[0].get_facecolors(), expected_colors)

    def test_series_color_no_index(self):
        colors_ord = pd.Series(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 'a'])
        ax1 = self.df.plot(colors_ord)
        self.df['colors_ord'] = colors_ord
        ax2 = self.df.plot('colors_ord')
        point_colors1 = ax1.collections[0].get_facecolors()
        point_colors2 = ax2.collections[0].get_facecolors()
        np.testing.assert_array_equal(point_colors1[1], point_colors2[1])

    def test_series_color_index(self):
        colors_ord = pd.Series(['a', 'a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'], index=[0, 3, 6, 9, 1, 4, 7, 2, 5, 8])
        ax1 = self.df.plot(colors_ord)
        self.df['colors_ord'] = colors_ord
        ax2 = self.df.plot('colors_ord')
        point_colors1 = ax1.collections[0].get_facecolors()
        point_colors2 = ax2.collections[0].get_facecolors()
        np.testing.assert_array_equal(point_colors1[1], point_colors2[1])

    def test_colormap(self):
        ax = self.points.plot(cmap='RdYlGn')
        cmap = plt.get_cmap('RdYlGn')
        exp_colors = cmap(np.arange(self.N) / (self.N - 1))
        _check_colors(self.N, ax.collections[0].get_facecolors(), exp_colors)
        ax = self.df.plot(cmap='RdYlGn')
        _check_colors(self.N, ax.collections[0].get_facecolors(), exp_colors)
        ax = self.df.plot(column='values', cmap='RdYlGn')
        cmap = plt.get_cmap('RdYlGn')
        _check_colors(self.N, ax.collections[0].get_facecolors(), exp_colors)
        ax = self.points.plot(cmap=plt.get_cmap('Set1', lut=5))
        cmap = plt.get_cmap('Set1', lut=5)
        exp_colors = cmap(list(range(5)) * 2)
        _check_colors(self.N, ax.collections[0].get_facecolors(), exp_colors)

    def test_single_color(self):
        ax = self.points.plot(color='green')
        _check_colors(self.N, ax.collections[0].get_facecolors(), ['green'] * self.N)
        ax = self.df.plot(color='green')
        _check_colors(self.N, ax.collections[0].get_facecolors(), ['green'] * self.N)
        ax = self.df.plot(color=(0.5, 0.5, 0.5))
        _check_colors(self.N, ax.collections[0].get_facecolors(), [(0.5, 0.5, 0.5)] * self.N)
        ax = self.df.plot(color=(0.5, 0.5, 0.5, 0.5))
        _check_colors(self.N, ax.collections[0].get_facecolors(), [(0.5, 0.5, 0.5, 0.5)] * self.N)
        with pytest.raises((ValueError, TypeError)):
            self.df.plot(color='not color')
        with warnings.catch_warnings(record=True) as _:
            ax = self.df.plot(column='values', color='green')
            _check_colors(self.N, ax.collections[0].get_facecolors(), ['green'] * self.N)

    def test_markersize(self):
        ax = self.points.plot(markersize=10)
        assert ax.collections[0].get_sizes() == [10]
        ax = self.df.plot(markersize=10)
        assert ax.collections[0].get_sizes() == [10]
        ax = self.df.plot(column='values', markersize=10)
        assert ax.collections[0].get_sizes() == [10]
        ax = self.df.plot(markersize='values')
        assert (ax.collections[0].get_sizes() == self.df['values']).all()
        ax = self.df.plot(column='values', markersize='values')
        assert (ax.collections[0].get_sizes() == self.df['values']).all()

    def test_markerstyle(self):
        ax = self.df2.plot(marker='+')
        expected = _style_to_vertices('+')
        np.testing.assert_array_equal(expected, ax.collections[0].get_paths()[0].vertices)

    def test_style_kwargs(self):
        ax = self.points.plot(edgecolors='k')
        assert (ax.collections[0].get_edgecolor() == [0, 0, 0, 1]).all()

    def test_style_kwargs_alpha(self):
        ax = self.df.plot(alpha=0.7)
        np.testing.assert_array_equal([0.7], ax.collections[0].get_alpha())
        try:
            ax = self.df.plot(alpha=np.linspace(0, 0.0, 1.0, self.N))
        except TypeError:
            pass
        else:
            np.testing.assert_array_equal(np.linspace(0, 0.0, 1.0, self.N), ax.collections[0].get_alpha())

    @pytest.mark.skipif(Version(matplotlib.__version__) >= Version('3.8.0.dev') and Version(matplotlib.__version__) < Version('3.8.0'), reason='failing with matplotlib dev')
    def test_legend(self):
        with warnings.catch_warnings(record=True) as _:
            ax = self.df.plot(column='values', color='green', legend=True)
            assert len(ax.get_figure().axes) == 1
        ax = self.df.plot(legend=True)
        assert len(ax.get_figure().axes) == 1
        ax = self.df.plot(column='values', cmap='RdYlGn', legend=True)
        point_colors = ax.collections[0].get_facecolors()
        cbar_colors = _get_colorbar_ax(ax.get_figure()).collections[-1].get_facecolors()
        np.testing.assert_array_equal(point_colors[0], cbar_colors[0])
        np.testing.assert_array_equal(point_colors[-1], cbar_colors[-1])
        ax = self.df.plot(column='values', categorical=True, legend=True)
        point_colors = ax.collections[0].get_facecolors()
        cbar_colors = ax.get_legend().axes.collections[-1].get_facecolors()
        np.testing.assert_array_equal(point_colors[0], cbar_colors[0])
        np.testing.assert_array_equal(point_colors[-1], cbar_colors[-1])
        norm = matplotlib.colors.LogNorm(vmin=self.df[1:].exp.min(), vmax=self.df[1:].exp.max())
        ax = self.df[1:].plot(column='exp', cmap='RdYlGn', legend=True, norm=norm)
        point_colors = ax.collections[0].get_facecolors()
        cbar_colors = _get_colorbar_ax(ax.get_figure()).collections[-1].get_facecolors()
        np.testing.assert_array_equal(point_colors[0], cbar_colors[0])
        np.testing.assert_array_equal(point_colors[-1], cbar_colors[-1])
        assert cbar_colors.shape == (256, 4)

    def test_subplots_norm(self):
        cmap = matplotlib.cm.viridis_r
        norm = matplotlib.colors.Normalize(vmin=0, vmax=20)
        ax = self.df.plot(column='values', cmap=cmap, norm=norm)
        actual_colors_orig = ax.collections[0].get_facecolors()
        exp_colors = cmap(np.arange(10) / 20)
        np.testing.assert_array_equal(exp_colors, actual_colors_orig)
        fig, ax = plt.subplots()
        self.df[1:].plot(column='values', ax=ax, norm=norm, cmap=cmap)
        actual_colors_sub = ax.collections[0].get_facecolors()
        np.testing.assert_array_equal(actual_colors_orig[1], actual_colors_sub[0])

    def test_empty_plot(self):
        s = GeoSeries([Polygon()])
        with pytest.warns(UserWarning):
            ax = s.plot()
        assert len(ax.collections) == 0
        s = GeoSeries([])
        with pytest.warns(UserWarning):
            ax = s.plot()
        assert len(ax.collections) == 0
        df = GeoDataFrame([], columns=['geometry'])
        with pytest.warns(UserWarning):
            ax = df.plot()
        assert len(ax.collections) == 0

    def test_empty_geometry(self):
        if compat.USE_PYGEOS:
            s = GeoSeries([wkt.loads('POLYGON EMPTY')])
            s = GeoSeries([Polygon([(0, 0), (1, 0), (1, 1)]), wkt.loads('POLYGON EMPTY')])
            ax = s.plot()
            assert len(ax.collections) == 1
        if not compat.USE_PYGEOS:
            s = GeoSeries([Polygon([(0, 0), (1, 0), (1, 1)]), Polygon()])
            ax = s.plot()
            assert len(ax.collections) == 1
        poly = Polygon([(-1, -1), (-1, 2), (2, 2), (2, -1), (-1, -1)])
        point = Point(0, 1)
        point_ = Point(10, 10)
        empty_point = Point()
        gdf = GeoDataFrame(geometry=[point, empty_point, point_])
        gdf['geometry'] = gdf.intersection(poly)
        gdf.loc[3] = [None]
        ax = gdf.plot()
        assert len(ax.collections) == 1

    @pytest.mark.parametrize('geoms', [[box(0, 0, 1, 1), box(7, 7, 8, 8)], [LineString([(1, 1), (1, 2)]), LineString([(7, 1), (7, 2)])], [Point(1, 1), Point(7, 7)]])
    def test_empty_geometry_colors(self, geoms):
        s = GeoSeries(geoms, index=['r', 'b'])
        s2 = s.intersection(box(5, 0, 10, 10))
        ax = s2.plot(color=['red', 'blue'])
        blue = np.array([0.0, 0.0, 1.0, 1.0])
        if s.geom_type['r'] == 'LineString':
            np.testing.assert_array_equal(ax.get_children()[0].get_edgecolor()[0], blue)
        else:
            np.testing.assert_array_equal(ax.get_children()[0].get_facecolor()[0], blue)

    def test_multipoints(self):
        ax = self.df2.plot()
        _check_colors(4, ax.collections[0].get_facecolors(), [MPL_DFT_COLOR] * 4)
        ax = self.df2.plot(column='values')
        cmap = plt.get_cmap(lut=2)
        expected_colors = [cmap(0)] * self.N + [cmap(1)] * self.N
        _check_colors(20, ax.collections[0].get_facecolors(), expected_colors)
        ax = self.df2.plot(color=['r', 'b'])
        _check_colors(20, ax.collections[0].get_facecolors(), ['r'] * 10 + ['b'] * 10)

    def test_multipoints_alpha(self):
        ax = self.df2.plot(alpha=0.7)
        np.testing.assert_array_equal([0.7], ax.collections[0].get_alpha())
        try:
            ax = self.df2.plot(alpha=[0.7, 0.2])
        except TypeError:
            pass
        else:
            np.testing.assert_array_equal([0.7] * 10 + [0.2] * 10, ax.collections[0].get_alpha())

    def test_categories(self):
        self.df['cats_object'] = ['cat1', 'cat2'] * 5
        self.df['nums'] = [1, 2] * 5
        self.df['singlecat_object'] = ['cat2'] * 10
        self.df['cats'] = pd.Categorical(['cat1', 'cat2'] * 5)
        self.df['singlecat'] = pd.Categorical(['cat2'] * 10, categories=['cat1', 'cat2'])
        self.df['cats_ordered'] = pd.Categorical(['cat2', 'cat1'] * 5, categories=['cat2', 'cat1'])
        self.df['bool'] = [False, True] * 5
        self.df['bool_extension'] = pd.array([False, True] * 5)
        self.df['cats_string'] = pd.array(['cat1', 'cat2'] * 5, dtype='string')
        ax1 = self.df.plot('cats_object', legend=True)
        ax2 = self.df.plot('cats', legend=True)
        ax3 = self.df.plot('singlecat_object', categories=['cat1', 'cat2'], legend=True)
        ax4 = self.df.plot('singlecat', legend=True)
        ax5 = self.df.plot('cats_ordered', legend=True)
        ax6 = self.df.plot('nums', categories=[1, 2], legend=True)
        ax7 = self.df.plot('bool', legend=True)
        ax8 = self.df.plot('bool_extension', legend=True)
        ax9 = self.df.plot('cats_string', legend=True)
        point_colors1 = ax1.collections[0].get_facecolors()
        for ax in [ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
            point_colors2 = ax.collections[0].get_facecolors()
            np.testing.assert_array_equal(point_colors1[1], point_colors2[1])
        legend1 = [x.get_markerfacecolor() for x in ax1.get_legend().get_lines()]
        for ax in [ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
            legend2 = [x.get_markerfacecolor() for x in ax.get_legend().get_lines()]
            np.testing.assert_array_equal(legend1, legend2)
        with pytest.raises(TypeError):
            self.df.plot(column='cats_object', categories='non_list')
        with pytest.raises(ValueError, match='Column contains values not listed in categories.'):
            self.df.plot(column='cats_object', categories=['cat1'])
        with pytest.raises(ValueError, match="Cannot specify 'categories' when column has"):
            self.df.plot(column='cats', categories=['cat1'])

    def test_missing(self):
        self.df.loc[0, 'values'] = np.nan
        ax = self.df.plot('values')
        cmap = plt.get_cmap()
        expected_colors = cmap(np.arange(self.N - 1) / (self.N - 2))
        _check_colors(self.N - 1, ax.collections[0].get_facecolors(), expected_colors)
        ax = self.df.plot('values', missing_kwds={'color': 'r'})
        cmap = plt.get_cmap()
        expected_colors = cmap(np.arange(self.N - 1) / (self.N - 2))
        _check_colors(1, ax.collections[1].get_facecolors(), ['r'])
        _check_colors(self.N - 1, ax.collections[0].get_facecolors(), expected_colors)
        ax = self.df.plot('values', missing_kwds={'color': 'r'}, categorical=True, legend=True)
        _check_colors(1, ax.collections[1].get_facecolors(), ['r'])
        point_colors = ax.collections[0].get_facecolors()
        nan_color = ax.collections[1].get_facecolors()
        leg_colors = ax.get_legend().axes.collections[0].get_facecolors()
        leg_colors1 = ax.get_legend().axes.collections[1].get_facecolors()
        np.testing.assert_array_equal(point_colors[0], leg_colors[0])
        np.testing.assert_array_equal(nan_color[0], leg_colors1[0])

    def test_no_missing_and_missing_kwds(self):
        df = self.df.copy()
        df['category'] = df['values'].astype('str')
        df.plot('category', missing_kwds={'facecolor': 'none'}, legend=True)