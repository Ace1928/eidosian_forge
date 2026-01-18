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
def update_curvature(self, curv=None):
    """Update the curvature in the MinModeAtoms object."""
    if curv:
        self.curvature = curv
    else:
        self.curvature = np.vdot(self.forces2 - self.forces1, self.eigenmode) / (2.0 * self.dR)