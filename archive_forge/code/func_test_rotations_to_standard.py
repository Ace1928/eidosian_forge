from random import randint
import numpy as np
import pytest
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
from ase.utils.structure_comparator import SpgLibNotFoundError
from ase.build import bulk
from ase import Atoms
from ase.spacegroup import spacegroup, crystal
def test_rotations_to_standard(comparator):
    s1 = Atoms('Al')
    tol = 1e-06
    num_tests = 4
    if heavy_test:
        num_tests = 20
    for _ in range(num_tests):
        cell = np.random.rand(3, 3) * 4.0 - 4.0
        s1.set_cell(cell)
        new_cell = comparator._standarize_cell(s1).get_cell().T
        assert abs(new_cell[1, 0]) < tol
        assert abs(new_cell[2, 0]) < tol
        assert abs(new_cell[2, 1]) < tol