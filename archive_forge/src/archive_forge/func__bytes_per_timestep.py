import numpy as np
from itertools import islice
import os
from ase.atoms import Atoms, Atom
from ase.cell import Cell
from ase.io.formats import index2range
from ase.data import atomic_numbers
def _bytes_per_timestep(natoms):
    return 4 + 6 * 8 + 7 * 4 + 3 * 4 * natoms