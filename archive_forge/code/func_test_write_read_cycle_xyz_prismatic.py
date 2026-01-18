import numpy as np
import pytest
from ase import Atoms
from ase.io import read
from ase.build import bulk
from .test_mustem import make_STO_atoms
def test_write_read_cycle_xyz_prismatic():
    """Check writing and reading a xtl mustem file."""
    atoms = bulk('Si', cubic=True)
    atoms.set_array('occupancies', np.ones(len(atoms)))
    rng = np.random.RandomState(42)
    atoms.set_array('debye_waller_factors', 0.62 + 0.1 * rng.rand(len(atoms)))
    filename = 'SI100.XYZ'
    atoms.write(filename=filename, format='prismatic', comments='one unit cell of 100 silicon')
    atoms_loaded = read(filename=filename, format='prismatic')
    np.testing.assert_allclose(atoms.positions, atoms_loaded.positions)
    np.testing.assert_allclose(atoms.cell, atoms_loaded.cell)
    np.testing.assert_allclose(atoms.get_array('occupancies'), atoms_loaded.get_array('occupancies'))
    np.testing.assert_allclose(atoms.get_array('debye_waller_factors'), atoms_loaded.get_array('debye_waller_factors'), rtol=1e-05)