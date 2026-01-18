import os
import numpy as np
from ase.units import Bohr, Ha, Ry, fs, m, s
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.calculators.openmx.reader import (read_electron_valency, get_file_name, get_standard_key)
from ase.calculators.openmx import parameters as param
def write_tuple_float(fd, key, value):
    fd.write('        '.join([key, '%.4f %.4f %.4f' % value]))
    fd.write('\n')