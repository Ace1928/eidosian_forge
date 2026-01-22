import sys
import weakref
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
def _getmasses(self):
    """Get the masses as an Nx1 array."""
    return np.reshape(self.atoms.get_masses(), (-1, 1))