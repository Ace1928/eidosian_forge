from math import pi, sqrt, cos
import pytest
import numpy as np
from ase import Atoms
from ase import data
from ase.lattice.cubic import FaceCenteredCubic
def test_vacuum_one_direction(atoms):
    vac = 10.0
    atoms.center(axis=2, vacuum=vac)
    c = atoms.get_cell()
    checkang(c[0], c[1], pi / 2)
    checkang(c[0], c[2], pi / 2)
    checkang(c[1], c[2], pi / 2)
    assert np.abs(4.5 * a0 + 2 * vac - c[2, 2]) < 1e-10