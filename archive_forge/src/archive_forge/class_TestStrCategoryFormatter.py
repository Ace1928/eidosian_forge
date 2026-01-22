import warnings
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.category as cat
from matplotlib.testing.decorators import check_figures_equal
class TestStrCategoryFormatter:
    test_cases = [('ascii', ['hello', 'world', 'hi']), ('unicode', ['Здравствуйте', 'привет'])]
    ids, cases = zip(*test_cases)

    @pytest.mark.parametrize('ydata', cases, ids=ids)
    def test_StrCategoryFormatter(self, ydata):
        unit = cat.UnitData(ydata)
        labels = cat.StrCategoryFormatter(unit._mapping)
        for i, d in enumerate(ydata):
            assert labels(i, i) == d
            assert labels(i, None) == d

    @pytest.mark.parametrize('ydata', cases, ids=ids)
    @pytest.mark.parametrize('plotter', PLOT_LIST, ids=PLOT_IDS)
    def test_StrCategoryFormatterPlot(self, ydata, plotter):
        ax = plt.figure().subplots()
        plotter(ax, range(len(ydata)), ydata)
        for i, d in enumerate(ydata):
            assert ax.yaxis.major.formatter(i) == d
        assert ax.yaxis.major.formatter(i + 1) == ''