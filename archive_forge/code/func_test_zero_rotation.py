import pytest
import numpy as np
from ase.build import molecule
from ase.md.velocitydistribution import Stationary, ZeroRotation
def test_zero_rotation(atoms):
    mom1 = atoms.get_angular_momentum()
    ZeroRotation(atoms)
    mom2 = atoms.get_angular_momentum()
    assert norm(mom1) > 0.1
    assert norm(mom2) < 1e-13