import time
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.calculators.lammpsrun import Prism
from ase.neighborlist import NeighborList
from ase.data import atomic_masses, chemical_symbols
from ase.io import read
def write_lammps_definitions(self, atoms, btypes, atypes, dtypes):
    """Write force field definitions for LAMMPS."""
    with open(self.prefix + '_opls', 'w') as fd:
        self._write_lammps_definitions(fd, atoms, btypes, atypes, dtypes)