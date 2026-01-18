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
def update_virtual_forces(self, extrapolated_forces=False):
    """Get the forces at the endpoints of the dimer."""
    self.update_virtual_positions()
    if extrapolated_forces:
        self.forces1 = self.forces1E.copy()
    else:
        self.forces1 = self.atoms.get_forces(real=True, pos=self.pos1)
    if self.control.get_parameter('use_central_forces'):
        self.forces2 = 2 * self.forces0 - self.forces1
    else:
        self.forces2 = self.atoms.get_forces(real=True, pos=self.pos2)