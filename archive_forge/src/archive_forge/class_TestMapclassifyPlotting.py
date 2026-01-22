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
class TestMapclassifyPlotting:

    @classmethod
    def setup_class(cls):
        try:
            import mapclassify
        except ImportError:
            pytest.importorskip('mapclassify')
        cls.mc = mapclassify
        cls.classifiers = list(mapclassify.classifiers.CLASSIFIERS)
        cls.classifiers.remove('UserDefined')
        pth = get_path('naturalearth_lowres')
        cls.df = read_file(pth)
        cls.df['NEGATIVES'] = np.linspace(-10, 10, len(cls.df.index))
        cls.df['low_vals'] = np.linspace(0, 0.3, cls.df.shape[0])
        cls.df['mid_vals'] = np.linspace(0.3, 0.7, cls.df.shape[0])
        cls.df['high_vals'] = np.linspace(0.7, 1.0, cls.df.shape[0])
        cls.df.loc[cls.df.index[:20:2], 'high_vals'] = np.nan
        cls.nybb = read_file(get_path('nybb'))
        cls.nybb['vals'] = [0.001, 0.002, 0.003, 0.004, 0.005]

    def test_legend(self):
        with warnings.catch_warnings(record=True) as _:
            ax = self.df.plot(column='pop_est', scheme='QUANTILES', k=3, cmap='OrRd', legend=True)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        expected = [s.split('|')[0][1:-2] for s in str(self.mc.Quantiles(self.df['pop_est'], k=3)).split('\n')[4:]]
        assert labels == expected

    def test_bin_labels(self):
        ax = self.df.plot(column='pop_est', scheme='QUANTILES', k=3, cmap='OrRd', legend=True, legend_kwds={'labels': ['foo', 'bar', 'baz']})
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        expected = ['foo', 'bar', 'baz']
        assert labels == expected

    def test_invalid_labels_length(self):
        with pytest.raises(ValueError):
            self.df.plot(column='pop_est', scheme='QUANTILES', k=3, cmap='OrRd', legend=True, legend_kwds={'labels': ['foo', 'bar']})

    def test_negative_legend(self):
        ax = self.df.plot(column='NEGATIVES', scheme='FISHER_JENKS', k=3, cmap='OrRd', legend=True)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        expected = ['-10.00,  -3.41', ' -3.41,   3.30', '  3.30,  10.00']
        assert labels == expected

    def test_fmt(self):
        ax = self.df.plot(column='NEGATIVES', scheme='FISHER_JENKS', k=3, cmap='OrRd', legend=True, legend_kwds={'fmt': '{:.0f}'})
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        expected = ['-10,  -3', ' -3,   3', '  3,  10']
        assert labels == expected

    def test_interval(self):
        ax = self.df.plot(column='NEGATIVES', scheme='FISHER_JENKS', k=3, cmap='OrRd', legend=True, legend_kwds={'interval': True})
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        expected = ['[-10.00,  -3.41]', '( -3.41,   3.30]', '(  3.30,  10.00]']
        assert labels == expected

    @pytest.mark.parametrize('scheme', ['FISHER_JENKS', 'FISHERJENKS'])
    def test_scheme_name_compat(self, scheme):
        ax = self.df.plot(column='NEGATIVES', scheme=scheme, k=3, legend=True)
        assert len(ax.get_legend().get_texts()) == 3

    def test_schemes(self):
        for scheme in self.classifiers:
            self.df.plot(column='pop_est', scheme=scheme, legend=True)

    def test_classification_kwds(self):
        ax = self.df.plot(column='pop_est', scheme='percentiles', k=3, classification_kwds={'pct': [50, 100]}, cmap='OrRd', legend=True)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        expected = [s.split('|')[0][1:-2] for s in str(self.mc.Percentiles(self.df['pop_est'], pct=[50, 100])).split('\n')[4:]]
        assert labels == expected

    def test_invalid_scheme(self):
        with pytest.raises(ValueError):
            scheme = 'invalid_scheme_*#&)(*#'
            self.df.plot(column='gdp_md_est', scheme=scheme, k=3, cmap='OrRd', legend=True)

    def test_cax_legend_passing(self):
        """Pass a 'cax' argument to 'df.plot(.)', that is valid only if 'ax' is
        passed as well (if not, a new figure is created ad hoc, and 'cax' is
        ignored)
        """
        ax = plt.axes()
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        with pytest.raises(ValueError):
            ax = self.df.plot(column='pop_est', cmap='OrRd', legend=True, cax=cax)

    def test_cax_legend_height(self):
        """Pass a cax argument to 'df.plot(.)', the legend location must be
        aligned with those of main plot
        """
        with warnings.catch_warnings(record=True) as _:
            ax = self.df.plot(column='pop_est', cmap='OrRd', legend=True)
        plot_height = _get_ax(ax.get_figure(), '').get_position().height
        legend_height = _get_ax(ax.get_figure(), '<colorbar>').get_position().height
        assert abs(plot_height - legend_height) >= 1e-06
        fig, ax2 = plt.subplots()
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.1, label='fixed_colorbar')
        with warnings.catch_warnings(record=True) as _:
            ax2 = self.df.plot(column='pop_est', cmap='OrRd', legend=True, cax=cax, ax=ax2)
        plot_height = _get_ax(fig, '').get_position().height
        legend_height = _get_ax(fig, 'fixed_colorbar').get_position().height
        assert abs(plot_height - legend_height) < 1e-06

    def test_empty_bins(self):
        bins = np.arange(1, 11) / 10
        ax = self.df.plot('low_vals', scheme='UserDefined', classification_kwds={'bins': bins}, legend=True)
        expected = np.array([[0.281412, 0.155834, 0.469201, 1.0], [0.267004, 0.004874, 0.329415, 1.0], [0.244972, 0.287675, 0.53726, 1.0]])
        assert all(((z == expected).all(axis=1).any() for z in ax.collections[0].get_facecolors()))
        labels = ['0.00, 0.10', '0.10, 0.20', '0.20, 0.30', '0.30, 0.40', '0.40, 0.50', '0.50, 0.60', '0.60, 0.70', '0.70, 0.80', '0.80, 0.90', '0.90, 1.00']
        legend = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == legend
        legend_colors_exp = [(0.267004, 0.004874, 0.329415, 1.0), (0.281412, 0.155834, 0.469201, 1.0), (0.244972, 0.287675, 0.53726, 1.0), (0.190631, 0.407061, 0.556089, 1.0), (0.147607, 0.511733, 0.557049, 1.0), (0.119699, 0.61849, 0.536347, 1.0), (0.20803, 0.718701, 0.472873, 1.0), (0.430983, 0.808473, 0.346476, 1.0), (0.709898, 0.868751, 0.169257, 1.0), (0.993248, 0.906157, 0.143936, 1.0)]
        assert [line.get_markerfacecolor() for line in ax.get_legend().get_lines()] == legend_colors_exp
        ax2 = self.df.plot('mid_vals', scheme='UserDefined', classification_kwds={'bins': bins}, legend=True)
        expected = np.array([[0.244972, 0.287675, 0.53726, 1.0], [0.190631, 0.407061, 0.556089, 1.0], [0.147607, 0.511733, 0.557049, 1.0], [0.119699, 0.61849, 0.536347, 1.0], [0.20803, 0.718701, 0.472873, 1.0]])
        assert all(((z == expected).all(axis=1).any() for z in ax2.collections[0].get_facecolors()))
        labels = ['-inf, 0.10', '0.10, 0.20', '0.20, 0.30', '0.30, 0.40', '0.40, 0.50', '0.50, 0.60', '0.60, 0.70', '0.70, 0.80', '0.80, 0.90', '0.90, 1.00']
        legend = [t.get_text() for t in ax2.get_legend().get_texts()]
        assert labels == legend
        assert [line.get_markerfacecolor() for line in ax2.get_legend().get_lines()] == legend_colors_exp
        ax3 = self.df.plot('high_vals', scheme='UserDefined', classification_kwds={'bins': bins}, legend=True)
        expected = np.array([[0.709898, 0.868751, 0.169257, 1.0], [0.993248, 0.906157, 0.143936, 1.0], [0.430983, 0.808473, 0.346476, 1.0]])
        assert all(((z == expected).all(axis=1).any() for z in ax3.collections[0].get_facecolors()))
        legend = [t.get_text() for t in ax3.get_legend().get_texts()]
        assert labels == legend
        assert [line.get_markerfacecolor() for line in ax3.get_legend().get_lines()] == legend_colors_exp

    def test_equally_formatted_bins(self):
        ax = self.nybb.plot('vals', scheme='quantiles', legend=True)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        expected = ['0.00, 0.00', '0.00, 0.00', '0.00, 0.00', '0.00, 0.00', '0.00, 0.01']
        assert labels == expected
        ax2 = self.nybb.plot('vals', scheme='quantiles', legend=True, legend_kwds={'fmt': '{:.3f}'})
        labels = [t.get_text() for t in ax2.get_legend().get_texts()]
        expected = ['0.001, 0.002', '0.002, 0.003', '0.003, 0.003', '0.003, 0.004', '0.004, 0.005']
        assert labels == expected