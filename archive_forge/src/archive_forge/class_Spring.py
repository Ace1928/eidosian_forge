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
class Spring:

    def __init__(self, atoms1, atoms2, energy1, energy2, k):
        self.atoms1 = atoms1
        self.atoms2 = atoms2
        self.energy1 = energy1
        self.energy2 = energy2
        self.k = k

    def _find_mic(self):
        pos1 = self.atoms1.get_positions()
        pos2 = self.atoms2.get_positions()
        mic, _ = find_mic(pos2 - pos1, self.atoms1.cell, self.atoms1.pbc)
        return mic

    @lazyproperty
    def t(self):
        return self._find_mic()

    @lazyproperty
    def nt(self):
        return np.linalg.norm(self.t)