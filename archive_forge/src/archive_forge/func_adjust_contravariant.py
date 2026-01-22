from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
def adjust_contravariant(self, vecs, B):
    """Adjust the values of a set of vectors that are contravariant with the unit transformation"""
    vecs = self.Jacobian_inv @ (vecs.flatten() - B)
    vecs = (self.Jacobian @ vecs + B).reshape((-1, 3))
    return vecs