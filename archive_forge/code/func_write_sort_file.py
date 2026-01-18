import os
import warnings
import shutil
from os.path import join, isfile, islink
from typing import List, Sequence, Tuple
import numpy as np
import ase
from ase.calculators.calculator import kpts2ndarray
from ase.calculators.vasp.setups import get_default_setups
def write_sort_file(self, directory='./'):
    """Writes a sortings file.

        This file contains information about how the atoms are sorted in
        the first column and how they should be resorted in the second
        column. It is used for restart purposes to get sorting right
        when reading in an old calculation to ASE."""
    file = open(join(directory, 'ase-sort.dat'), 'w')
    for n in range(len(self.sort)):
        file.write('%5i %5i \n' % (self.sort[n], self.resort[n]))