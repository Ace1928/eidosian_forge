import os
import time
import math
import itertools
import numpy as np
from scipy.spatial.distance import cdist
from ase.io import write, read
from ase.geometry.cell import cell_to_cellpar
from ase.data import covalent_radii
from ase.ga import get_neighbor_list
class CellBounds:
    """Class for defining as well as checking limits on
    cell vector lengths and angles.

    Parameters:

    bounds: dict
        Any of the following keywords can be used, in
        conjunction with a [low, high] list determining
        the lower and upper bounds:

        a, b, c:
           Minimal and maximal lengths (in Angstrom)
           for the 1st, 2nd and 3rd lattice vectors.

        alpha, beta, gamma:
           Minimal and maximal values (in degrees)
           for the angles between the lattice vectors.

        phi, chi, psi:
           Minimal and maximal values (in degrees)
           for the angles between each lattice vector
           and the plane defined by the other two vectors.

    Example:

    >>> from ase.ga.utilities import CellBounds
    >>> CellBounds(bounds={'phi': [20, 160],
    ...                    'chi': [60, 120],
    ...                    'psi': [20, 160],
    ...                    'a': [2, 20], 'b': [2, 20], 'c': [2, 20]})
    """

    def __init__(self, bounds={}):
        self.bounds = {'alpha': [0, np.pi], 'beta': [0, np.pi], 'gamma': [0, np.pi], 'phi': [0, np.pi], 'chi': [0, np.pi], 'psi': [0, np.pi], 'a': [0, 1000000.0], 'b': [0, 1000000.0], 'c': [0, 1000000.0]}
        for param, bound in bounds.items():
            if param not in ['a', 'b', 'c']:
                bound = [b * np.pi / 180.0 for b in bound]
            self.bounds[param] = bound

    def is_within_bounds(self, cell):
        values = get_cell_angles_lengths(cell)
        verdict = True
        for param, bound in self.bounds.items():
            if not bound[0] <= values[param] <= bound[1]:
                verdict = False
        return verdict