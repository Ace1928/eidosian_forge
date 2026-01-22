import sys
import weakref
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
def _getnatoms(self):
    """Get the number of atoms.

        In a parallel simulation, this is the total number of atoms on all
        processors.
        """
    return len(self.atoms)