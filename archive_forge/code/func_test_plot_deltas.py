from collections import OrderedDict
import numpy as np
import pytest
from typing import List, Tuple, Any
from ase.spectrum.dosdata import DOSData, GridDOSData, RawDOSData
@pytest.mark.usefixtures('figure')
@pytest.mark.parametrize('linewidth', linewidths)
def test_plot_deltas(self, sparse_dos, figure, linewidth):
    if linewidth is None:
        mplargs = None
    else:
        mplargs = {'linewidth': linewidth}
    ax = figure.add_subplot(111)
    ax_out = sparse_dos.plot_deltas(ax=ax, mplargs=mplargs)
    assert ax_out == ax
    assert np.allclose(list(map(lambda x: x.vertices, ax.get_children()[0].get_paths())), [[[1.2, 0.0], [1.2, 3.0]], [[3.4, 0.0], [3.4, 2.1]], [[5.0, 0.0], [5.0, 0.0]]])