from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
class FixCartesian(FixConstraintSingle):
    """Fix an atom index *a* in the directions of the cartesian coordinates."""

    def __init__(self, a, mask=(1, 1, 1)):
        self.a = a
        self.mask = ~np.asarray(mask, bool)

    def get_removed_dof(self, atoms):
        return 3 - self.mask.sum()

    def adjust_positions(self, atoms, new):
        step = new[self.a] - atoms.positions[self.a]
        step *= self.mask
        new[self.a] = atoms.positions[self.a] + step

    def adjust_forces(self, atoms, forces):
        forces[self.a] *= self.mask

    def __repr__(self):
        return 'FixCartesian(a={0}, mask={1})'.format(self.a, list(~self.mask))

    def todict(self):
        return {'name': 'FixCartesian', 'kwargs': {'a': self.a, 'mask': ~self.mask.tolist()}}