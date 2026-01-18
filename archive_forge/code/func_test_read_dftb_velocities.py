import numpy as np
from io import StringIO
from ase.atoms import Atoms
from ase.units import AUT, Bohr, second
from ase.io.dftb import (read_dftb, read_dftb_lattice,
def test_read_dftb_velocities():
    atoms = Atoms('H2')
    filename = 'geo_end.xyz'
    with open(filename, 'w') as fd:
        fd.write(geo_end_xyz)
    read_dftb_velocities(atoms, filename=filename)
    velocities = np.linspace(-1, 2, num=6).reshape(2, 3)
    velocities /= 1e-12 * second
    assert np.allclose(velocities, atoms.get_velocities())