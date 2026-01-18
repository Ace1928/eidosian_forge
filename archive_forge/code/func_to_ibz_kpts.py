import re
from collections import OrderedDict
import numpy as np
from ase import Atoms
from ase.units import Hartree, Bohr
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
from .parser import _define_pattern
def to_ibz_kpts(self):
    if not self.ibz_kpts:
        return np.array([[0.0, 0.0, 0.0]])
    sorted_kpts = sorted(list(self.ibz_kpts.items()), key=lambda x: x[0])
    return np.array(list(zip(*sorted_kpts))[1])