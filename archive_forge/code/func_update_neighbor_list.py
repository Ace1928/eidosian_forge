import time
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.calculators.lammpsrun import Prism
from ase.neighborlist import NeighborList
from ase.data import atomic_masses, chemical_symbols
from ase.io import read
def update_neighbor_list(self, atoms):
    cut = 0.5 * max(self.data['cutoffs'].values())
    self.nl = NeighborList([cut] * len(atoms), skin=0, bothways=True, self_interaction=False)
    self.nl.update(atoms)
    self.atoms = atoms