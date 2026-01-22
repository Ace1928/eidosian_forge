from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
class FixAngleCombo(FixInternalsBase):
    """Constraint subobject for fixing linear combination of angles
        within FixInternals.

        sum_i( coef_i * angle_i ) = constant
        """

    def gather_vectors(self, pos):
        v0 = [pos[h] - pos[k] for h, k, l in self.indices]
        v1 = [pos[l] - pos[k] for h, k, l in self.indices]
        return (v0, v1)

    def prepare_jacobian(self, pos):
        v0, v1 = self.gather_vectors(pos)
        derivs = get_angles_derivatives(v0, v1, cell=self.cell, pbc=self.pbc)
        self.finalize_jacobian(pos, len(v0), 3, derivs)

    def adjust_positions(self, oldpos, newpos):
        v0, v1 = self.gather_vectors(newpos)
        value = get_angles(v0, v1, cell=self.cell, pbc=self.pbc)
        value = np.dot(self.coefs, value)
        self.sigma = value - self.targetvalue
        self.finalize_positions(newpos)

    def __repr__(self):
        return 'FixAngleCombo({}, {}, {})'.format(self.targetvalue, self.indices, self.coefs)