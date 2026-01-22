import numpy as np
import numpy.testing as nptest
from numpy.testing import assert_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics import gofplots
from statsmodels.graphics.gofplots import (
from statsmodels.graphics.utils import _import_mpl
class TestQQLine:

    def setup_method(self):
        np.random.seed(0)
        self.x = np.sort(np.random.normal(loc=2.9, scale=1.2, size=37))
        self.y = np.sort(np.random.normal(loc=3.0, scale=1.1, size=37))
        try:
            import matplotlib.pyplot as plt
            self.fig, self.ax = plt.subplots()
            self.ax.plot(self.x, self.y, 'ko')
        except ImportError:
            pass
        self.lineoptions = {'linewidth': 2, 'dashes': (10, 1, 3, 4), 'color': 'green'}
        self.fmt = 'bo-'

    @pytest.mark.matplotlib
    def test_badline(self):
        with pytest.raises(ValueError):
            qqline(self.ax, 'junk')

    @pytest.mark.matplotlib
    def test_non45_no_x(self, close_figures):
        with pytest.raises(ValueError):
            qqline(self.ax, 's', y=self.y)

    @pytest.mark.matplotlib
    def test_non45_no_y(self, close_figures):
        with pytest.raises(ValueError):
            qqline(self.ax, 's', x=self.x)

    @pytest.mark.matplotlib
    def test_non45_no_x_no_y(self, close_figures):
        with pytest.raises(ValueError):
            qqline(self.ax, 's')

    @pytest.mark.matplotlib
    def test_45(self, close_figures):
        nchildren = len(self.ax.get_children())
        qqline(self.ax, '45')
        assert len(self.ax.get_children()) > nchildren

    @pytest.mark.matplotlib
    def test_45_fmt(self, close_figures):
        qqline(self.ax, '45', fmt=self.fmt)

    @pytest.mark.matplotlib
    def test_45_fmt_lineoptions(self, close_figures):
        qqline(self.ax, '45', fmt=self.fmt, **self.lineoptions)

    @pytest.mark.matplotlib
    def test_r(self, close_figures):
        nchildren = len(self.ax.get_children())
        qqline(self.ax, 'r', x=self.x, y=self.y)
        assert len(self.ax.get_children()) > nchildren

    @pytest.mark.matplotlib
    def test_r_fmt(self, close_figures):
        qqline(self.ax, 'r', x=self.x, y=self.y, fmt=self.fmt)

    @pytest.mark.matplotlib
    def test_r_fmt_lineoptions(self, close_figures):
        qqline(self.ax, 'r', x=self.x, y=self.y, fmt=self.fmt, **self.lineoptions)

    @pytest.mark.matplotlib
    def test_s(self, close_figures):
        nchildren = len(self.ax.get_children())
        qqline(self.ax, 's', x=self.x, y=self.y)
        assert len(self.ax.get_children()) > nchildren

    @pytest.mark.matplotlib
    def test_s_fmt(self, close_figures):
        qqline(self.ax, 's', x=self.x, y=self.y, fmt=self.fmt)

    @pytest.mark.matplotlib
    def test_s_fmt_lineoptions(self, close_figures):
        qqline(self.ax, 's', x=self.x, y=self.y, fmt=self.fmt, **self.lineoptions)

    @pytest.mark.matplotlib
    def test_q(self, close_figures):
        nchildren = len(self.ax.get_children())
        qqline(self.ax, 'q', dist=stats.norm, x=self.x, y=self.y)
        assert len(self.ax.get_children()) > nchildren

    @pytest.mark.matplotlib
    def test_q_fmt(self, close_figures):
        qqline(self.ax, 'q', dist=stats.norm, x=self.x, y=self.y, fmt=self.fmt)

    @pytest.mark.matplotlib
    def test_q_fmt_lineoptions(self, close_figures):
        qqline(self.ax, 'q', dist=stats.norm, x=self.x, y=self.y, fmt=self.fmt, **self.lineoptions)