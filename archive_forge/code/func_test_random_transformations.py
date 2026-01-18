import pytest
import numpy as np
from ase.build import bulk, make_supercell
from ase.lattice import FCC, BCC
from ase.calculators.emt import EMT
def test_random_transformations(atoms):
    rng = np.random.RandomState(44)
    e0 = getenergy(atoms)
    imgs = []
    i = 0
    while i < 10:
        P = rng.randint(-2, 3, size=(3, 3))
        detP = np.linalg.det(P)
        if detP == 0:
            continue
        elif detP < 0:
            P[0] *= -1
        bigatoms = make_supercell(atoms, P)
        imgs.append(bigatoms)
        getenergy(bigatoms, eref=e0)
        i += 1