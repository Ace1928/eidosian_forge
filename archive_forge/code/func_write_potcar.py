import os
import warnings
import shutil
from os.path import join, isfile, islink
from typing import List, Sequence, Tuple
import numpy as np
import ase
from ase.calculators.calculator import kpts2ndarray
from ase.calculators.vasp.setups import get_default_setups
def write_potcar(self, suffix='', directory='./'):
    """Writes the POTCAR file."""
    with open(join(directory, 'POTCAR' + suffix), 'w') as potfile:
        for filename in self.ppp_list:
            with open_potcar(filename=filename) as ppp_file:
                for line in ppp_file:
                    potfile.write(line)