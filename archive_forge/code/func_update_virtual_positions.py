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
def update_virtual_positions(self):
    """Update the end point positions."""
    self.pos1 = self.pos0 + self.eigenmode * self.dR
    self.pos2 = self.pos0 - self.eigenmode * self.dR