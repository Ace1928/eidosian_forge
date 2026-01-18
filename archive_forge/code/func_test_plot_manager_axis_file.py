import os
import numpy as np
import pytest
from ase.lattice.cubic import FaceCenteredCubic
from ase.utils.plotting import SimplePlottingAxes
from ase.visualize.plot import plot_atoms
def test_plot_manager_axis_file(self, testdir, xy_data, figure):
    filename = 'plot.png'
    x, y = xy_data
    ax = figure.add_subplot(111)
    with SimplePlottingAxes(ax=ax, show=False, filename=filename) as return_ax:
        assert return_ax is ax
        ax.plot(x, y)
    assert os.path.isfile(filename)