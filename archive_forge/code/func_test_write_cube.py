import pytest
import numpy as np
from ase.transport.tools import dagger, normalize
from ase.dft.kpoints import monkhorst_pack
from ase.build import molecule
from ase.io.cube import read_cube
from ase.lattice import CUB, FCC, BCC, TET, BCT, ORC, ORCF, ORCI, ORCC, HEX, \
from ase.dft.wannier import gram_schmidt, lowdin, random_orthogonal_matrix, \
def test_write_cube(wan):
    atoms = molecule('H2')
    atoms.center(vacuum=3.0)
    wanf = wan(atoms=atoms)
    index = 0
    cubefilename = 'wanf.cube'
    wanf.write_cube(index, cubefilename)
    with open(cubefilename, mode='r') as inputfile:
        content = read_cube(inputfile)
    assert pytest.approx(content['atoms'].cell.array) == atoms.cell.array
    assert pytest.approx(content['data']) == wanf.get_function(index)