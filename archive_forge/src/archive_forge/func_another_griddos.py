import pytest
from typing import Iterable
import numpy as np
from ase.spectrum.doscollection import (DOSCollection,
from ase.spectrum.dosdata import DOSData, RawDOSData, GridDOSData
@pytest.fixture
def another_griddos(self):
    energies = np.linspace(1, 10, 7)
    weights = np.cos(energies)
    return GridDOSData(energies, weights, info={'my_key': 'other_value'})