import numpy as np
from ase import Atoms
def test_positions(atoms=atoms):
    assert np.allclose(positions, atoms.get_positions())