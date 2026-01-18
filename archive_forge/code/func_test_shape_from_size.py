import numpy as np
from ase import Atoms
from ase.calculators.calculator import kpts2sizeandoffsets as k2so
def test_shape_from_size():
    size, offsets = map(tuple, k2so(size=(3, 4, 5)))
    assert (size, offsets) == ((3, 4, 5), (0, 0, 0))