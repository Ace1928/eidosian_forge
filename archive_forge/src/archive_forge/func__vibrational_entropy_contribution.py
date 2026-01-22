import os
import sys
import numpy as np
from ase import units
def _vibrational_entropy_contribution(self, temperature):
    """Calculates the entropy due to vibrations for a set of vibrations
        given in eV and a temperature given in Kelvin.  Returns the entropy
        in eV/K."""
    kT = units.kB * temperature
    S_v = 0.0
    for energy in self.vib_energies:
        x = energy / kT
        S_v += x / (np.exp(x) - 1.0) - np.log(1.0 - np.exp(-x))
    S_v *= units.kB
    return S_v