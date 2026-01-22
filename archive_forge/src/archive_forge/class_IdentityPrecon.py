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
class IdentityPrecon(Precon):
    """
    Dummy preconditioner which does not modify forces
    """

    def make_precon(self, atoms, reinitialize=None):
        self.atoms = atoms

    def Pdot(self, x):
        return x

    def solve(self, x):
        return x

    def copy(self):
        return IdentityPrecon()

    def asarray(self):
        return np.eye(3 * len(self.atoms))