from random import randint
import numpy as np
import pytest
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
from ase.utils.structure_comparator import SpgLibNotFoundError
from ase.build import bulk
from ase import Atoms
from ase.spacegroup import spacegroup, crystal
def test_original_paper_structures():
    syms = ['O', 'O', 'Mg', 'F']
    cell1 = [(3.16, 0.0, 0.0), (-0.95, 4.14, 0.0), (-0.95, -0.22, 4.13)]
    p1 = [(0.44, 0.4, 0.3), (0.94, 0.4, 0.79), (0.45, 0.9, 0.79), (0.94, 0.4, 0.29)]
    s1 = Atoms(syms, cell=cell1, scaled_positions=p1, pbc=True)
    cell2 = [(6.0, 0.0, 0.0), (1.0, 3.0, 0.0), (2.0, -3.0, 3.0)]
    p2 = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0)]
    s2 = Atoms(syms, cell=cell2, scaled_positions=p2, pbc=True)
    comp = SymmetryEquivalenceCheck()
    assert comp.compare(s1, s2)
    assert comp.compare(s2, s1) == comp.compare(s1, s2)