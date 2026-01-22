import os
import numpy as np
from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from ase.units import Bohr, Hartree
def adp_dipole(self, r, rvec, d):
    mu = np.sum(rvec * d(r)[:, np.newaxis], axis=0)
    return mu