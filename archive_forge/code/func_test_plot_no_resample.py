import pytest
from typing import Iterable
import numpy as np
from ase.spectrum.doscollection import (DOSCollection,
from ase.spectrum.dosdata import DOSData, RawDOSData, GridDOSData
@pytest.mark.usefixtures('figure')
def test_plot_no_resample(self, griddoscollection, figure):
    ax = figure.add_subplot(111)
    griddoscollection.plot(ax=ax)
    assert np.allclose(ax.get_lines()[0].get_xdata(), griddoscollection[0].get_energies())
    assert np.allclose(ax.get_lines()[1].get_ydata(), griddoscollection[1].get_weights())