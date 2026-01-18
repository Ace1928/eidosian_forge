import optparse
import numpy as np
from ase.data import covalent_radii
from ase.io.cube import read_cube_data
from ase.data.colors import cpk_colors
from ase.calculators.calculator import get_calculator_class
Plot atoms, unit-cell and iso-surfaces using Mayavi.

    Parameters:

    atoms: Atoms object
        Positions, atomiz numbers and unit-cell.
    data: 3-d ndarray of float
        Data for iso-surfaces.
    countours: list of float
        Contour values.
    