import numpy as np
from ase.ga import get_raw_score
class EnergyComparator:
    """Compares the energy of the supplied atoms objects using
       get_potential_energy().

       Parameters:

       dE: the difference in energy below which two energies are
       deemed equal.
    """

    def __init__(self, dE=0.02):
        self.dE = dE

    def looks_like(self, a1, a2):
        dE = abs(a1.get_potential_energy() - a2.get_potential_energy())
        if dE >= self.dE:
            return False
        else:
            return True