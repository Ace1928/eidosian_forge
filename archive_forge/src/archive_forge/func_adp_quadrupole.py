import os
import numpy as np
from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from ase.units import Bohr, Hartree
def adp_quadrupole(self, r, rvec, q):
    r = np.sqrt(np.sum(rvec ** 2, axis=1))
    lam = np.zeros([rvec.shape[0], 3, 3])
    qr = q(r)
    for alpha in range(3):
        for beta in range(3):
            lam[:, alpha, beta] += qr * rvec[:, alpha] * rvec[:, beta]
    return np.sum(lam, axis=0)