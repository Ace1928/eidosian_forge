from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
class FixCartesianParametricRelations(FixParametricRelations):

    def __init__(self, indices, Jacobian, const_shift, params=None, eps=1e-12, use_cell=False):
        """The Cartesian coordinate version of FixParametricRelations"""
        super(FixCartesianParametricRelations, self).__init__(indices, Jacobian, const_shift, params, eps, use_cell)

    def adjust_contravariant(self, vecs, B):
        """Adjust the values of a set of vectors that are contravariant with the unit transformation"""
        vecs = self.Jacobian_inv @ (vecs.flatten() - B)
        vecs = (self.Jacobian @ vecs + B).reshape((-1, 3))
        return vecs

    def adjust_positions(self, atoms, positions):
        """Adjust positions of the atoms to match the constraints"""
        if self.use_cell:
            return
        positions[self.indices] = self.adjust_contravariant(positions[self.indices], self.const_shift)

    def adjust_momenta(self, atoms, momenta):
        """Adjust momenta of the atoms to match the constraints"""
        if self.use_cell:
            return
        momenta[self.indices] = self.adjust_contravariant(momenta[self.indices], np.zeros(self.const_shift.shape))

    def adjust_forces(self, atoms, forces):
        """Adjust forces of the atoms to match the constraints"""
        if self.use_cell:
            return
        forces_reduced = self.Jacobian.T @ forces[self.indices].flatten()
        forces[self.indices] = (self.Jacobian_inv.T @ forces_reduced).reshape(-1, 3)

    def adjust_cell(self, atoms, cell):
        """Adjust the cell of the atoms to match the constraints"""
        if not self.use_cell:
            return
        cell[self.indices] = self.adjust_contravariant(cell[self.indices], np.zeros(self.const_shift.shape))

    def adjust_stress(self, atoms, stress):
        """Adjust the stress of the atoms to match the constraints"""
        if not self.use_cell:
            return
        stress_3x3 = voigt_6_to_full_3x3_stress(stress)
        stress_reduced = self.Jacobian.T @ stress_3x3[self.indices].flatten()
        stress_3x3[self.indices] = (self.Jacobian_inv.T @ stress_reduced).reshape(-1, 3)
        stress[:] = full_3x3_to_voigt_6_stress(stress_3x3)