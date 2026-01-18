import numpy as np
from io import StringIO
from ase.atoms import Atoms
from ase.units import AUT, Bohr, second
from ase.io.dftb import (read_dftb, read_dftb_lattice,
def test_read_dftb_lattice():
    vectors = read_dftb_lattice(fd_lattice)
    mols = [Atoms(), Atoms()]
    read_dftb_lattice(fd_lattice, mols)
    compareVec = np.array([[26.1849388999576, 5.773808884828536e-06, 9.076696618724854e-06], [0.115834159141441, 26.1947703089401, 9.372892011565608e-06], [0.635711495837792, 0.451552307731081, 9.42069476334197]])
    assert (vectors[0] == compareVec).all()
    assert len(vectors) == 2
    assert len(vectors[1]) == 3
    assert (mols[0].get_cell() == compareVec).all()
    assert mols[1].get_pbc().all()