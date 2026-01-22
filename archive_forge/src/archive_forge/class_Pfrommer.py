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
class Pfrommer(Precon):
    """
    Use initial guess for inverse Hessian from Pfrommer et al. as a
    simple preconditioner

    J. Comput. Phys. vol 131 p233-240 (1997)
    """

    def __init__(self, bulk_modulus=500 * units.GPa, phonon_frequency=50 * THz, apply_positions=True, apply_cell=True):
        """
        Default bulk modulus is 500 GPa and default phonon frequency is 50 THz
        """
        self.bulk_modulus = bulk_modulus
        self.phonon_frequency = phonon_frequency
        self.apply_positions = apply_positions
        self.apply_cell = apply_cell
        self.H0 = None

    def make_precon(self, atoms, reinitialize=None):
        if self.H0 is not None:
            return
        variable_cell = False
        if isinstance(atoms, Filter):
            variable_cell = True
            atoms = atoms.atoms
        omega = self.phonon_frequency
        mass = atoms.get_masses().mean()
        block = np.eye(3) / (mass * omega ** 2)
        blocks = [block] * len(atoms)
        if variable_cell:
            coeff = 1.0
            if self.apply_cell:
                coeff = 1.0 / (3 * self.bulk_modulus)
            blocks.append(np.diag([coeff] * 9))
        self.H0 = sparse.block_diag(blocks, format='csr')
        return

    def Pdot(self, x):
        return self.H0.solve(x)

    def solve(self, x):
        y = self.H0.dot(x)
        return y

    def copy(self):
        return Pfrommer(self.bulk_modulus, self.phonon_frequency, self.apply_positions, self.apply_cell)

    def asarray(self):
        return np.array(self.H0.todense())