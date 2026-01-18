import itertools
import numpy as np
from ase.utils import pbc2pbc
from ase.cell import Cell
Calculate a Minkowski-reduced lattice basis.  The reduced basis
    has the shortest possible vector lengths and has
    norm(a) <= norm(b) <= norm(c).

    Implements the method described in:

    Low-dimensional Lattice Basis Reduction Revisited
    Nguyen, Phong Q. and Stehlé, Damien,
    ACM Trans. Algorithms 5(4) 46:1--46:48, 2009
    https://doi.org/10.1145/1597036.1597050

    Parameters:

    cell: array
        The lattice basis to reduce (in row-vector format).
    pbc: array, optional
        The periodic boundary conditions of the cell (Default `True`).
        If `pbc` is provided, only periodic cell vectors are reduced.

    Returns:

    rcell: array
        The reduced lattice basis.
    op: array
        The unimodular matrix transformation (rcell = op @ cell).
    