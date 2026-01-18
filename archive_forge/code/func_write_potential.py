import os
import numpy as np
from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from ase.units import Bohr, Hartree
def write_potential(self, filename, nc=1, numformat='%.8e'):
    """Writes out the potential in the format given by the form
        variable to 'filename' with a data format that is nc columns
        wide.  Note: array lengths need to be an exact multiple of nc
        """
    with open(filename, 'wb') as fd:
        self._write_potential(fd, nc=nc, numformat=numformat)