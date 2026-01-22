import os
import warnings
import numpy as np
import ase
from ase.data import atomic_masses
from ase.geometry import cellpar_to_cell
import collections
from functools import reduce
def _set_atoms(self, atoms=None):
    """
        Associate an Atoms object with the trajectory.

        For internal use only.
        """
    if atoms is not None and (not hasattr(atoms, 'get_positions')):
        raise TypeError('"atoms" argument is not an Atoms object.')
    self.atoms = atoms