import numpy as np
import numpy.testing as nptest
from numpy.testing import assert_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics import gofplots
from statsmodels.graphics.gofplots import (
from statsmodels.graphics.utils import _import_mpl
class TestDoPlot:

    def setup_method(self):
        try:
            import matplotlib.pyplot as plt
            self.fig, self.ax = plt.subplots()
        except ImportError:
            pass
        self.x = [0.2, 0.6, 2.0, 4.5, 10.0, 50.0, 83.0, 99.1, 99.7]
        self.y = [1.2, 1.4, 1.7, 2.1, 3.2, 3.7, 4.5, 5.1, 6.3]
        self.full_options = {'marker': 's', 'markerfacecolor': 'cornflowerblue', 'markeredgecolor': 'firebrick', 'markeredgewidth': 1.25, 'linestyle': '--'}
        self.step_options = {'linestyle': '-', 'where': 'mid'}

    @pytest.mark.matplotlib
    def test_baseline(self, close_figures):
        plt = _import_mpl()
        fig, ax = gofplots._do_plot(self.x, self.y)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert self.fig is not fig
        assert self.ax is not ax

    @pytest.mark.matplotlib
    def test_with_ax(self, close_figures):
        plt = _import_mpl()
        fig, ax = gofplots._do_plot(self.x, self.y, ax=self.ax)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        assert self.fig is fig
        assert self.ax is ax

    @pytest.mark.matplotlib
    def test_plot_full_options(self, close_figures):
        gofplots._do_plot(self.x, self.y, ax=self.ax, step=False, **self.full_options)

    @pytest.mark.matplotlib
    def test_step_baseline(self, close_figures):
        gofplots._do_plot(self.x, self.y, ax=self.ax, step=True, **self.step_options)

    @pytest.mark.matplotlib
    def test_step_full_options(self, close_figures):
        gofplots._do_plot(self.x, self.y, ax=self.ax, step=True, **self.full_options)

    @pytest.mark.matplotlib
    def test_plot_qq_line(self, close_figures):
        gofplots._do_plot(self.x, self.y, ax=self.ax, line='r')

    @pytest.mark.matplotlib
    def test_step_qq_line(self, close_figures):
        gofplots._do_plot(self.x, self.y, ax=self.ax, step=True, line='r')