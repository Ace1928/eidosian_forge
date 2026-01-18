import numpy as np
from ase.build import bulk
from ase.io.aims import read_aims as read
from ase.io.aims import parse_geometry_lines
from pytest import approx
def test_scaled(atoms=atoms):
    """write fractional coords and check if structure was preserved"""
    atoms.write(file, format=format, scaled=True, wrap=False)
    new_atoms = read(file)
    assert np.allclose(atoms.positions, new_atoms.positions), (atoms.positions, new_atoms.positions)