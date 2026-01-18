import pytest
import numpy as np
from ase.build import bulk
def test_set_abc(atoms, displacement, reference):
    for i, atom in enumerate(atoms):
        atom.a += displacement[i, 0]
        atom.b += displacement[i, 1]
        atom.c += displacement[i, 2]
    assert np.allclose(get_spos(atoms), reference)