import os
import numpy as np
import pytest
from ase.lattice.cubic import FaceCenteredCubic
from ase.utils.plotting import SimplePlottingAxes
from ase.visualize.plot import plot_atoms
@pytest.fixture
def xy_data(self):
    return ([1, 2], [3, 4])