import re
import os
import numpy as np
import ase
from .vasp import Vasp
from ase.calculators.singlepoint import SinglePointCalculator
def _read_chg(self, fobj, chg, volume):
    """Read charge from file object

        Utility method for reading the actual charge density (or
        charge density difference) from a file object. On input, the
        file object must be at the beginning of the charge block, on
        output the file position will be left at the end of the
        block. The chg array must be of the correct dimensions.

        """
    for zz in range(chg.shape[2]):
        for yy in range(chg.shape[1]):
            chg[:, yy, zz] = np.fromfile(fobj, count=chg.shape[0], sep=' ')
    chg /= volume