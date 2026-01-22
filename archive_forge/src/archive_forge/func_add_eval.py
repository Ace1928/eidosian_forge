import re
from collections import OrderedDict
import numpy as np
from ase import Atoms
from ase.units import Hartree, Bohr
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
from .parser import _define_pattern
def add_eval(self, index, spin, energy, occ):
    if index not in self.data:
        self.data[index] = dict()
    if spin not in self.data[index]:
        self.data[index][spin] = []
    self.data[index][spin].append((energy, occ))