import pytest
from ase import Atoms
from ase.md.contour_exploration import ContourExploration
import numpy as np
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
def Al_atom_pair(pair_distance=pair_distance):
    atoms = Atoms('AlAl', positions=[[-pair_distance / 2, 0, 0], [pair_distance / 2, 0, 0]])
    atoms.center(vacuum=10)
    atoms.calc = EMT()
    return atoms