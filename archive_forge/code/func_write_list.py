import os
from os.path import join
import re
from glob import glob
import warnings
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.units import Hartree, Bohr, fs
from ase.calculators.calculator import Parameters
def write_list(fd, value, unit):
    for element in value:
        fd.write('{} '.format(element))
    if unit is not None:
        fd.write('{}'.format(unit))
    fd.write('\n')