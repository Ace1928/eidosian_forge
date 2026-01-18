import numpy as np
from ase.calculators.calculator import Calculator
from ase.calculators.calculator import compare_atoms
from . import kimpy_wrappers
from . import neighborlist
@property
def update_neigh(self):
    return self.neigh.update