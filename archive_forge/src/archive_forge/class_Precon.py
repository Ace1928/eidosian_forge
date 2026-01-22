import sys
import time
import copy
import warnings
from abc import ABC, abstractmethod
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline
from ase.constraints import Filter, FixAtoms
from ase.utils import longsum
from ase.geometry import find_mic
import ase.utils.ff as ff
import ase.units as units
from ase.optimize.precon.neighbors import (get_neighbours,
from ase.neighborlist import neighbor_list
class Precon(ABC):

    @abstractmethod
    def make_precon(self, atoms, reinitialize=None):
        """
        Create a preconditioner matrix based on the passed set of atoms.

        Creates a general-purpose preconditioner for use with optimization
        algorithms, based on examining distances between pairs of atoms in the
        lattice. The matrix will be stored in the attribute self.P and
        returned.

        Args:
            atoms: the Atoms object used to create the preconditioner.
            
            reinitialize: if True, parameters of the preconditioner
                will be recalculated before the preconditioner matrix is
                created. If False, they will be calculated only when they
                do not currently have a value (ie, the first time this
                function is called).

        Returns:
            P: A sparse scipy csr_matrix. BE AWARE that using
                numpy.dot() with sparse matrices will result in
                errors/incorrect results - use the .dot method directly
                on the matrix instead.
        """
        ...

    @abstractmethod
    def Pdot(self, x):
        """
        Return the result of applying P to a vector x
        """
        ...

    def dot(self, x, y):
        """
        Return the preconditioned dot product <P x, y>

        Uses 128-bit floating point math for vector dot products
        """
        return longsum(self.Pdot(x) * y)

    def norm(self, x):
        """
        Return the P-norm of x, where |x|_P = sqrt(<Px, x>)
        """
        return np.sqrt(self.dot(x, x))

    def vdot(self, x, y):
        return self.dot(x.reshape(-1), y.reshape(-1))

    @abstractmethod
    def solve(self, x):
        """
        Solve the (sparse) linear system P x = y and return y
        """
        ...

    def apply(self, forces, atoms):
        """
        Convenience wrapper that combines make_precon() and solve()

        Parameters
        ----------
        forces: array
            (len(atoms)*3) array of input forces
        atoms: ase.atoms.Atoms

        Returns
        -------
        precon_forces: array
            (len(atoms), 3) array of preconditioned forces
        residual: float
            inf-norm of original forces, i.e. maximum absolute force
        """
        self.make_precon(atoms)
        residual = np.linalg.norm(forces, np.inf)
        precon_forces = self.solve(forces)
        return (precon_forces, residual)

    @abstractmethod
    def copy(self):
        ...

    @abstractmethod
    def asarray(self):
        """
        Array representation of preconditioner, as a dense matrix
        """
        ...