import numpy as np
from ase import Atoms
from ase.io import read, write
import ase.io.rmc6f as rmc6f
from ase.lattice.compounds import TRI_Fe2O3
def test_rmc6f_write_with_order():
    """Test for writing rmc6f input file with order passed in."""
    tol = 1e-05
    write('output.rmc6f', rmc6f_atoms, order=['F', 'S'])
    readback = read('output.rmc6f')
    reordered_positions = np.vstack((rmc6f_atoms.positions[1:7], rmc6f_atoms.positions[0]))
    assert np.allclose(reordered_positions, readback.positions, rtol=tol)