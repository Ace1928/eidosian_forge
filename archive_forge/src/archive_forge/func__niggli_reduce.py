from collections import Counter
from itertools import combinations, product, filterfalse
import numpy as np
from scipy.spatial import cKDTree as KDTree
from ase import Atom, Atoms
from ase.build.tools import niggli_reduce
def _niggli_reduce(self, atoms):
    """Reduce to niggli cells.

        Reduce the atoms to niggli cells, then rotates the niggli cells to
        the so called "standard" orientation with one lattice vector along the
        x-axis and a second vector in the xy plane.
        """
    niggli_reduce(atoms)
    self._standarize_cell(atoms)