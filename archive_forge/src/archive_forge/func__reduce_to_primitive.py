from collections import Counter
from itertools import combinations, product, filterfalse
import numpy as np
from scipy.spatial import cKDTree as KDTree
from ase import Atom, Atoms
from ase.build.tools import niggli_reduce
def _reduce_to_primitive(self, structure):
    """Reduce the two structure to their primitive type"""
    try:
        import spglib
    except ImportError:
        raise SpgLibNotFoundError('SpgLib is required if to_primitive=True')
    cell = structure.get_cell().tolist()
    pos = structure.get_scaled_positions().tolist()
    numbers = structure.get_atomic_numbers()
    cell, scaled_pos, numbers = spglib.standardize_cell((cell, pos, numbers), to_primitive=True)
    atoms = Atoms(scaled_positions=scaled_pos, numbers=numbers, cell=cell, pbc=True)
    return atoms