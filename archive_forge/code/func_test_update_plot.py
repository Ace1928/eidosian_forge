import warnings
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.category as cat
from matplotlib.testing.decorators import check_figures_equal
@pytest.mark.parametrize('plotter', PLOT_LIST, ids=PLOT_IDS)
def test_update_plot(self, plotter):
    ax = plt.figure().subplots()
    plotter(ax, ['a', 'b'], ['e', 'g'])
    plotter(ax, ['a', 'b', 'd'], ['f', 'a', 'b'])
    plotter(ax, ['b', 'c', 'd'], ['g', 'e', 'd'])
    axis_test(ax.xaxis, ['a', 'b', 'd', 'c'])
    axis_test(ax.yaxis, ['e', 'g', 'f', 'a', 'b', 'd'])