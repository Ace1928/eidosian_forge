from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
class MirrorForce(FixConstraint):
    """Constraint object for mirroring the force between two atoms.

    This class is designed to find a transition state with the help of a
    single optimization. It can be used if the transition state belongs to a
    bond breaking reaction. First the given bond length will be fixed until
    all other degrees of freedom are optimized, then the forces of the two
    atoms will be mirrored to find the transition state. The mirror plane is
    perpendicular to the connecting line of the atoms. Transition states in
    dependence of the force can be obtained by stretching the molecule and
    fixing its total length with *FixBondLength* or by using *ExternalForce*
    during the optimization with *MirrorForce*.

    Parameters
    ----------
    a1: int
        First atom index.
    a2: int
        Second atom index.
    max_dist: float
        Upper limit of the bond length interval where the transition state
        can be found.
    min_dist: float
        Lower limit of the bond length interval where the transition state
        can be found.
    fmax: float
        Maximum force used for the optimization.

    Notes
    -----
    You can combine this constraint for example with FixBondLength but make
    sure that *MirrorForce* comes first in the list if there are overlaps
    between atom1-2 and atom3-4:

    >>> con1 = MirrorForce(atom1, atom2)
    >>> con2 = FixBondLength(atom3, atom4)
    >>> atoms.set_constraint([con1, con2])

    """

    def __init__(self, a1, a2, max_dist=2.5, min_dist=1.0, fmax=0.1):
        self.indices = [a1, a2]
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.fmax = fmax

    def adjust_positions(self, atoms, new):
        pass

    def adjust_forces(self, atoms, forces):
        dist = np.subtract.reduce(atoms.positions[self.indices])
        d = np.linalg.norm(dist)
        if d < self.min_dist or d > self.max_dist:
            forces[:] *= 0
            return
        dist /= d
        df = np.subtract.reduce(forces[self.indices])
        f = df.dot(dist)
        con_saved = atoms.constraints
        try:
            con = [con for con in con_saved if not isinstance(con, MirrorForce)]
            atoms.set_constraint(con)
            forces_copy = atoms.get_forces()
        finally:
            atoms.set_constraint(con_saved)
        df1 = -1 / 2.0 * f * dist
        forces_copy[self.indices] += (df1, -df1)
        if (forces_copy ** 2).sum(axis=1).max() < self.fmax ** 2:
            factor = 1.0
        else:
            factor = 0.0
        df1 = -(1 + factor) / 2.0 * f * dist
        forces[self.indices] += (df1, -df1)

    def index_shuffle(self, atoms, ind):
        """Shuffle the indices of the two atoms in this constraint

        """
        newa = [-1, -1]
        for new, old in slice2enlist(ind, len(atoms)):
            for i, a in enumerate(self.indices):
                if old == a:
                    newa[i] = new
        if newa[0] == -1 or newa[1] == -1:
            raise IndexError('Constraint not part of slice')
        self.indices = newa

    def __repr__(self):
        return 'MirrorForce(%d, %d, %f, %f, %f)' % (self.indices[0], self.indices[1], self.max_dist, self.min_dist, self.fmax)

    def todict(self):
        return {'name': 'MirrorForce', 'kwargs': {'a1': self.indices[0], 'a2': self.indices[1], 'max_dist': self.max_dist, 'min_dist': self.min_dist, 'fmax': self.fmax}}