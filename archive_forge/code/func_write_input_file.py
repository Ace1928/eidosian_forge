import os
import re
import numpy as np
from ase import Atoms
from ase.io import read
from ase.io.dmol import write_dmol_car, write_dmol_incoor
from ase.units import Hartree, Bohr
from ase.calculators.calculator import FileIOCalculator, Parameters, ReadError
def write_input_file(self):
    """ Writes the input file. """
    with open(self.label + '.input', 'w') as fd:
        self._write_input_file(fd)