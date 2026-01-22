import sys
import weakref
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
def _getbox(self):
    """Get the computational box."""
    return self.atoms.get_cell()