import time
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.calculators.lammpsrun import Prism
from ase.neighborlist import NeighborList
from ase.data import atomic_masses, chemical_symbols
from ase.io import read
def write_lammps(self, atoms, prefix='lammps'):
    """Write input for a LAMMPS calculation."""
    self.prefix = prefix
    if hasattr(atoms, 'connectivities'):
        connectivities = atoms.connectivities
    else:
        btypes, blist = self.get_bonds(atoms)
        atypes, alist = self.get_angles()
        dtypes, dlist = self.get_dihedrals(alist, atypes)
        connectivities = {'bonds': blist, 'bond types': btypes, 'angles': alist, 'angle types': atypes, 'dihedrals': dlist, 'dihedral types': dtypes}
        self.write_lammps_definitions(atoms, btypes, atypes, dtypes)
        self.write_lammps_in()
    self.write_lammps_atoms(atoms, connectivities)