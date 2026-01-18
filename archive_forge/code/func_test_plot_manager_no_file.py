import os
import numpy as np
import pytest
from ase.lattice.cubic import FaceCenteredCubic
from ase.utils.plotting import SimplePlottingAxes
from ase.visualize.plot import plot_atoms
def test_plot_manager_no_file(self, xy_data, figure):
    x, y = xy_data
    with SimplePlottingAxes(ax=None, show=False, filename=None) as ax:
        ax.plot(x, y)
    assert np.allclose(ax.lines[0].get_xydata().transpose(), xy_data)