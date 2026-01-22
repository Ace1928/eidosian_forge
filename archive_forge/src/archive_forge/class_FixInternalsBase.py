from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
class FixInternalsBase:
    """Base class for subclasses of FixInternals."""

    def __init__(self, targetvalue, indices, masses, cell, pbc):
        self.targetvalue = targetvalue
        self.indices = [defin[0:-1] for defin in indices]
        self.coefs = np.asarray([defin[-1] for defin in indices])
        self.masses = masses
        self.jacobian = []
        self.sigma = 1.0
        self.projected_force = None
        self.cell = cell
        self.pbc = pbc

    def finalize_jacobian(self, pos, n_internals, n, derivs):
        """Populate jacobian with derivatives for `n_internals` defined
            internals. n = 2 (bonds), 3 (angles), 4 (dihedrals)."""
        jacobian = np.zeros((n_internals, *pos.shape))
        for i, idx in enumerate(self.indices):
            for j in range(n):
                jacobian[i, idx[j]] = derivs[i, j]
        jacobian = jacobian.reshape((n_internals, 3 * len(pos)))
        self.jacobian = self.coefs @ jacobian

    def finalize_positions(self, newpos):
        jacobian = self.jacobian / self.masses
        lamda = -self.sigma / np.dot(jacobian, self.jacobian)
        dnewpos = lamda * jacobian
        newpos += dnewpos.reshape(newpos.shape)

    def adjust_forces(self, positions, forces):
        self.projected_force = np.dot(self.jacobian, forces.ravel())
        self.jacobian /= np.linalg.norm(self.jacobian)