import sys
import threading
import warnings
from abc import ABC, abstractmethod
import time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import cumtrapz
import ase.parallel
from ase.build import minimize_rotation_and_translation
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import MDMin
from ase.optimize.optimize import Optimizer
from ase.optimize.sciopt import OptimizerConvergenceError
from ase.geometry import find_mic
from ase.utils import lazyproperty, deprecated
from ase.utils.forcecurve import fit_images
from ase.optimize.precon import Precon, PreconImages
from ase.optimize.ode import ode12r
class NEBState:

    def __init__(self, neb, images, energies):
        self.neb = neb
        self.images = images
        self.energies = energies

    def spring(self, i):
        return Spring(self.images[i], self.images[i + 1], self.energies[i], self.energies[i + 1], self.neb.k[i])

    @lazyproperty
    def imax(self):
        return 1 + np.argsort(self.energies[1:-1])[-1]

    @property
    def emax(self):
        return self.energies[self.imax]

    @lazyproperty
    def eqlength(self):
        images = self.images
        beeline = images[self.neb.nimages - 1].get_positions() - images[0].get_positions()
        beelinelength = np.linalg.norm(beeline)
        return beelinelength / (self.neb.nimages - 1)

    @lazyproperty
    def nimages(self):
        return len(self.images)

    @property
    def precon(self):
        return self.neb.precon