import numpy as np
from ase import Atoms
from ase.io import read, write
import ase.io.rmc6f as rmc6f
from ase.lattice.compounds import TRI_Fe2O3
def test_rmc6f_write_with_triclinic_system():
    """Test for writing rmc6f input file for triclinic system
    """
    tol = 1e-05
    fe4o6 = TRI_Fe2O3(symbol=('Fe', 'O'), latticeconstant={'a': 5.143, 'b': 5.383, 'c': 14.902, 'alpha': 90.391, 'beta': 90.014, 'gamma': 89.834}, size=(1, 1, 1))
    va = [5.143, 0.0, 0.0]
    vb = [0.015596, 5.382977, 0.0]
    vc = [-0.00364124, -0.101684, 14.901653]
    write('output.rmc6f', fe4o6)
    readback = read('output.rmc6f')
    assert np.allclose(fe4o6.positions, readback.positions, rtol=tol)
    assert np.allclose(va, readback.cell[0], rtol=tol)
    assert np.allclose(vb, readback.cell[1], rtol=tol)
    assert np.allclose(vc, readback.cell[2], rtol=tol)