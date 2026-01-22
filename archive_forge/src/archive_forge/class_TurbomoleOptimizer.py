import os
import re
import warnings
from subprocess import Popen, PIPE
from math import log10, floor
import numpy as np
from ase import Atoms
from ase.units import Ha, Bohr
from ase.io import read, write
from ase.calculators.calculator import FileIOCalculator
from ase.calculators.calculator import PropertyNotImplementedError, ReadError
class TurbomoleOptimizer:

    def __init__(self, atoms, calc):
        self.atoms = atoms
        self.calc = calc
        self.atoms.calc = self.calc

    def todict(self):
        return {'type': 'optimization', 'optimizer': 'TurbomoleOptimizer'}

    def run(self, fmax=None, steps=None):
        if fmax is not None:
            self.calc.parameters['force convergence'] = fmax
            self.calc.verify_parameters()
        if steps is not None:
            self.calc.parameters['geometry optimization iterations'] = steps
            self.calc.verify_parameters()
        self.calc.calculate()
        self.atoms.positions[:] = self.calc.atoms.positions
        self.calc.parameters['task'] = 'energy'