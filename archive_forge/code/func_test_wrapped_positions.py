import numpy as np
from ase import Atoms
def test_wrapped_positions(atoms=atoms):
    atoms.wrap()
    assert np.allclose(positions_wrapped, atoms.get_positions())