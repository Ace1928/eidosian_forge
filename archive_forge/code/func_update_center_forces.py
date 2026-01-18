import sys
import time
import warnings
from math import cos, sin, atan, tan, degrees, pi, sqrt
from typing import Dict, Any
import numpy as np
from ase.optimize.optimize import Optimizer
from ase.parallel import world
from ase.calculators.singlepoint import SinglePointCalculator
from ase.utils import IOContext
def update_center_forces(self):
    """Get the forces at the center of the dimer."""
    self.atoms.set_positions(self.pos0)
    self.forces0 = self.atoms.get_forces(real=True)
    self.energy0 = self.atoms.get_potential_energy()