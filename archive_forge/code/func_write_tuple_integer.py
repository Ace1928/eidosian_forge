import os
import numpy as np
from ase.units import Bohr, Ha, Ry, fs, m, s
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.calculators.openmx.reader import (read_electron_valency, get_file_name, get_standard_key)
from ase.calculators.openmx import parameters as param
def write_tuple_integer(fd, key, value):
    fd.write('        '.join([key, '%d %d %d' % value]))
    fd.write('\n')