import re
from collections import OrderedDict
import numpy as np
from ase import Atoms
from ase.units import Hartree, Bohr
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
from .parser import _define_pattern
def to_singlepointkpts(self):
    kpts = []
    for i, (index, spins) in enumerate(self.data.items()):
        weight = self.weights[index]
        for spin, (_, data) in enumerate(spins.items()):
            energies, occs = np.array(sorted(data, key=lambda x: x[0])).T
            kpts.append(SinglePointKPoint(weight, spin, i, energies, occs))
    return kpts