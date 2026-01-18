import pytest
import numpy as np
from ase.build import molecule
from ase.md.velocitydistribution import Stationary, ZeroRotation
def test_stationary(atoms, stationary_atoms):
    assert norm(atoms.get_momenta().sum(axis=0)) > 0.1
    assert norm(stationary_atoms.get_momenta().sum(axis=0)) < 1e-13