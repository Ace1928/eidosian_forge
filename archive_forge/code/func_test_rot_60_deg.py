from random import randint
import numpy as np
import pytest
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
from ase.utils.structure_comparator import SpgLibNotFoundError
from ase.build import bulk
from ase import Atoms
from ase.spacegroup import spacegroup, crystal
def test_rot_60_deg(comparator):
    s1 = get_atoms_with_mixed_elements()
    s2 = s1.copy()
    ca = np.cos(np.pi / 3.0)
    sa = np.sin(np.pi / 3.0)
    matrix = np.array([[ca, sa, 0.0], [-sa, ca, 0.0], [0.0, 0.0, 1.0]])
    s2.set_positions(matrix.dot(s2.get_positions().T).T)
    s2.set_cell(matrix.dot(s2.get_cell().T).T)
    assert comparator.compare(s1, s2)