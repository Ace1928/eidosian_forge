from itertools import count
import numpy as np
from ase import Atoms
from ase.units import invcm, Ha
from ase.data import atomic_masses
from ase.calculators.calculator import all_changes
from ase.calculators.morse import MorsePotential
from ase.calculators.excitation_list import Excitation, ExcitationList
class H2MorseExcitedStates(ExcitationList):
    """First singlet excited states of H2"""

    def __init__(self, nstates=3):
        """
        Parameters
        ----------
        nstates: int, 1 <= nstates <= 3
          Number of excited states to consider, default 3
        """
        self.nstates = nstates
        super().__init__()

    def overlap(self, ov_nn, other):
        return ov_nn[1:len(self) + 1, 1:len(self) + 1] * ov_nn[0, 0]

    @classmethod
    def read(cls, filename, nstates=3):
        """Read myself from a file"""
        exl = cls(nstates)
        with open(filename, 'r') as fd:
            exl.filename = filename
            n = int(fd.readline().split()[0])
            for i in range(min(n, exl.nstates)):
                exl.append(H2Excitation.fromstring(fd.readline()))
        return exl

    def write(self, fname):
        with open(fname, 'w') as fd:
            fd.write('{0}\n'.format(len(self)))
            for ex in self:
                fd.write(ex.outstring())