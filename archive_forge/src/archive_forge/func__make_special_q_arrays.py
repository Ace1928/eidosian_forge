import sys
import weakref
import numpy as np
from ase.md.md import MolecularDynamics
from ase import units
def _make_special_q_arrays(self):
    """Make the arrays used to store data about the atoms.

        In a parallel simulation, these are migrating arrays.  In a
        serial simulation they are ordinary Numeric arrays.
        """
    natoms = len(self.atoms)
    self.q = np.zeros((natoms, 3), float)
    self.q_past = np.zeros((natoms, 3), float)
    self.q_future = np.zeros((natoms, 3), float)