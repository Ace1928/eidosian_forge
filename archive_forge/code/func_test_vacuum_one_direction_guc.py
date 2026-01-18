from math import pi, sqrt, cos
import pytest
import numpy as np
from ase import Atoms
from ase import data
from ase.lattice.cubic import FaceCenteredCubic
def test_vacuum_one_direction_guc(atoms_guc):
    atoms = atoms_guc
    vac = 10.0
    atoms.center(axis=2, vacuum=vac)
    c = atoms.get_cell()
    checkang(c[0], c[1], pi / 2)
    checkang(c[0], c[2], pi / 4)
    checkang(c[1], c[2], pi / 2)
    assert np.abs(2 * a0 + 2 * vac - c[2, 2]) < 1e-10
    atoms.center()
    c = atoms.get_cell()
    checkang(c[0], c[1], pi / 2)
    checkang(c[0], c[2], pi / 4)
    checkang(c[1], c[2], pi / 2)
    assert np.abs(2 * a0 + 2 * vac - c[2, 2]) < 1e-10
    a2 = atoms.copy()
    vac = 4.0
    atoms.center(vacuum=vac)
    c = atoms.get_cell()
    checkang(c[0], c[1], pi / 2)
    checkang(c[0], c[2], pi / 4)
    checkang(c[1], c[2], pi / 2)
    assert np.abs(4.5 * a0 + 2 * vac - c[1, 1]) < 1e-10
    assert np.abs(2 * a0 + 2 * vac - c[2, 2]) < 1e-10
    for i in range(3):
        a2.center(vacuum=vac, axis=i)
    assert abs(atoms.positions - a2.positions).max() < 1e-12
    assert abs(atoms.cell - a2.cell).max() < 1e-12