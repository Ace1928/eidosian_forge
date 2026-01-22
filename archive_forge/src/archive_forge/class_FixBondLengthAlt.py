from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
class FixBondLengthAlt(FixBondCombo):
    """Constraint subobject for fixing bond length within FixInternals.
        Fix distance between atoms with indices a1, a2."""

    def __init__(self, targetvalue, indices, masses, cell, pbc):
        indices = [list(indices) + [1.0]]
        super().__init__(targetvalue, indices, masses, cell=cell, pbc=pbc)

    def __repr__(self):
        return 'FixBondLengthAlt({}, {})'.format(self.targetvalue, *self.indices)