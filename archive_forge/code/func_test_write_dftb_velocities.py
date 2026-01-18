import numpy as np
from io import StringIO
from ase.atoms import Atoms
from ase.units import AUT, Bohr, second
from ase.io.dftb import (read_dftb, read_dftb_lattice,
def test_write_dftb_velocities():
    atoms = Atoms('H2')
    velocities = np.linspace(-1, 2, num=6).reshape(2, 3)
    atoms.set_velocities(velocities)
    write_dftb_velocities(atoms, filename='velocities.txt')
    velocities = np.loadtxt('velocities.txt') * Bohr / AUT
    assert np.allclose(velocities, atoms.get_velocities())