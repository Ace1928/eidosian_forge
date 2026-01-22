from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
class FixAngle(FixAngleCombo):
    """Constraint object for fixing an angle within
        FixInternals using the SHAKE algorithm.

        SHAKE convergence is potentially problematic for angles very close to
        0 or 180 degrees as there is a singularity in the Cartesian derivative.
        """

    def __init__(self, targetvalue, indices, masses, cell, pbc):
        """Fix atom movement to construct a constant angle."""
        indices = [list(indices) + [1.0]]
        super().__init__(targetvalue, indices, masses, cell=cell, pbc=pbc)

    def __repr__(self):
        return 'FixAngle({}, {})'.format(self.targetvalue, *self.indices)