import pytest
import numpy as np
from ase.build import bulk
def test_set_scaled_position(atoms, displacement, reference):
    for i, atom in enumerate(atoms):
        atom.scaled_position += displacement[i]
    assert np.allclose(get_spos(atoms), reference)