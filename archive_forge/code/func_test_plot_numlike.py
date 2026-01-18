import warnings
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.category as cat
from matplotlib.testing.decorators import check_figures_equal
@pytest.mark.parametrize('plotter', PLOT_LIST, ids=PLOT_IDS)
@pytest.mark.parametrize('ndata', numlike_data, ids=numlike_ids)
def test_plot_numlike(self, plotter, ndata):
    ax = plt.figure().subplots()
    counts = np.array([4, 6, 5])
    plotter(ax, ndata, counts)
    axis_test(ax.xaxis, ndata)