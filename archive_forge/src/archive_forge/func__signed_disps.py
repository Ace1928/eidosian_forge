import numpy as np
import ase.units as u
from ase.vibrations.raman import Raman, RamanPhonons
from ase.vibrations.resonant_raman import ResonantRaman
from ase.calculators.excitation_list import polarizability
def _signed_disps(self, sign):
    for a, i in zip(self.myindices, self.myxyz):
        yield self._disp(a, i, sign)