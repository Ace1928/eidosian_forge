from random import randint
import numpy as np
import pytest
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
from ase.utils.structure_comparator import SpgLibNotFoundError
from ase.build import bulk
from ase import Atoms
from ase.spacegroup import spacegroup, crystal
def test_single_impurity(comparator):
    s1 = bulk('Al')
    s1 = s1 * (2, 2, 2)
    s1[0].symbol = 'Mg'
    s2 = bulk('Al')
    s2 = s2 * (2, 2, 2)
    s2[3].symbol = 'Mg'
    assert comparator.compare(s1, s2)