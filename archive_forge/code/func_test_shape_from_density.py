import numpy as np
from ase import Atoms
from ase.calculators.calculator import kpts2sizeandoffsets as k2so
def test_shape_from_density():
    kd = 25 / (2 * np.pi)
    size, offsets = map(tuple, k2so(density=kd, atoms=Atoms(cell=(a, a, a), pbc=True)))
    assert (size, offsets) == ((5, 5, 5), (0, 0, 0))