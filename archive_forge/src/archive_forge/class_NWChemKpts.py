import re
from collections import OrderedDict
import numpy as np
from ase import Atoms
from ase.units import Hartree, Bohr
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
from .parser import _define_pattern
class NWChemKpts:

    def __init__(self):
        self.data = dict()
        self.ibz_kpts = dict()
        self.weights = dict()

    def add_ibz_kpt(self, index, raw_kpt):
        kpt = np.array([float(x.strip('>')) for x in raw_kpt.split()[1:4]])
        self.ibz_kpts[index] = kpt

    def add_eval(self, index, spin, energy, occ):
        if index not in self.data:
            self.data[index] = dict()
        if spin not in self.data[index]:
            self.data[index][spin] = []
        self.data[index][spin].append((energy, occ))

    def set_weight(self, index, weight):
        self.weights[index] = weight

    def to_ibz_kpts(self):
        if not self.ibz_kpts:
            return np.array([[0.0, 0.0, 0.0]])
        sorted_kpts = sorted(list(self.ibz_kpts.items()), key=lambda x: x[0])
        return np.array(list(zip(*sorted_kpts))[1])

    def to_singlepointkpts(self):
        kpts = []
        for i, (index, spins) in enumerate(self.data.items()):
            weight = self.weights[index]
            for spin, (_, data) in enumerate(spins.items()):
                energies, occs = np.array(sorted(data, key=lambda x: x[0])).T
                kpts.append(SinglePointKPoint(weight, spin, i, energies, occs))
        return kpts