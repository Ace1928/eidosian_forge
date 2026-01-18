import itertools
import numpy as np
import pytest
from matplotlib.axes import Axes, SubplotBase
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal, image_comparison
@pytest.mark.parametrize('xlabel_position', ['bottom', 'top'])
@pytest.mark.parametrize('ylabel_position', ['left', 'right'])
def test_subplots_hide_axislabels(xlabel_position, ylabel_position):
    axs = plt.figure().subplots(3, 3, sharex=True, sharey=True)
    for (i, j), ax in np.ndenumerate(axs):
        ax.set(xlabel='foo', ylabel='bar')
        ax.xaxis.set_label_position(xlabel_position)
        ax.yaxis.set_label_position(ylabel_position)
        ax.label_outer()
        assert bool(ax.get_xlabel()) == (xlabel_position == 'bottom' and i == 2 or (xlabel_position == 'top' and i == 0))
        assert bool(ax.get_ylabel()) == (ylabel_position == 'left' and j == 0 or (ylabel_position == 'right' and j == 2))