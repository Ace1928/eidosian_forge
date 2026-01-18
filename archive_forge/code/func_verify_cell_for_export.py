import numpy as np
from math import sqrt
from itertools import islice
from ase.io.formats import string2index
from ase.utils import rotate
from ase.data import covalent_radii, atomic_numbers
from ase.data.colors import jmol_colors
def verify_cell_for_export(cell, check_orthorhombric=True):
    """Function to verify if the cell size is defined and if the cell is

    Parameters:

    cell: cell object
        cell to be checked.

    check_orthorhombric: bool
        If True, check if the cell is orthorhombric, raise an ``ValueError`` if
        the cell is orthorhombric. If False, doesn't check if the cell is
        orthorhombric.

    Raise a ``ValueError`` if the cell if not suitable for export to mustem xtl
    file or prismatic/computem xyz format:
        - if cell is not orthorhombic (only when check_orthorhombric=True)
        - if cell size is not defined
    """
    if check_orthorhombric and (not cell.orthorhombic):
        raise ValueError('To export to this format, the cell needs to be orthorhombic.')
    if cell.rank < 3:
        raise ValueError('To export to this format, the cell size needs to be set: current cell is {}.'.format(cell))