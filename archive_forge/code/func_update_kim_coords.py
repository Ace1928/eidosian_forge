import numpy as np
from ase.calculators.calculator import Calculator
from ase.calculators.calculator import compare_atoms
from . import kimpy_wrappers
from . import neighborlist
@property
def update_kim_coords(self):
    return self.neigh.update_kim_coords