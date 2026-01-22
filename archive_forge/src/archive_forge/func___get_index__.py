import numpy as np
from ase import Atoms
def __get_index__(self):
    v = self.rng.rand() * self.rho[-1]
    for i in range(len(self.rho)):
        if self.rho[i] > v:
            return i