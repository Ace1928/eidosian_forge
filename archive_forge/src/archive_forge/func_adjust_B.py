from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
def adjust_B(self, cell, positions):
    """Wraps the positions back to the unit cell and adjust B to keep track of this change"""
    fractional = cell.scaled_positions(positions)
    wrapped_fractional = fractional % 1.0 % 1.0
    self.const_shift += np.round(wrapped_fractional - fractional).flatten()
    return cell.cartesian_positions(wrapped_fractional)