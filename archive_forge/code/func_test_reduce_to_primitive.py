from random import randint
import numpy as np
import pytest
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
from ase.utils.structure_comparator import SpgLibNotFoundError
from ase.build import bulk
from ase import Atoms
from ase.spacegroup import spacegroup, crystal
def test_reduce_to_primitive(comparator):
    atoms1 = crystal(symbols=['V', 'Li', 'O'], basis=[(0.0, 0.0, 0.0), (0.333333, 0.666667, 0.0), (0.333333, 0.0, 0.25)], spacegroup=167, cellpar=[5.123, 5.123, 13.005, 90.0, 90.0, 120.0], size=[1, 1, 1], primitive_cell=False)
    atoms2 = crystal(symbols=['V', 'Li', 'O'], basis=[(0.0, 0.0, 0.0), (0.333333, 0.666667, 0.0), (0.333333, 0.0, 0.25)], spacegroup=167, cellpar=[5.123, 5.123, 13.005, 90.0, 90.0, 120.0], size=[1, 1, 1], primitive_cell=True)
    try:
        comparator.to_primitive = True
        assert comparator.compare(atoms1, atoms2)
    except SpgLibNotFoundError:
        pass
    comparator.to_primitive = False