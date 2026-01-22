import re
from collections import OrderedDict
import numpy as np
from ase import Atoms
from ase.units import Hartree, Bohr
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
from .parser import _define_pattern
def add_ibz_kpt(self, index, raw_kpt):
    kpt = np.array([float(x.strip('>')) for x in raw_kpt.split()[1:4]])
    self.ibz_kpts[index] = kpt