from copy import deepcopy
from os.path import isfile
from warnings import warn
from numpy import array
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.parallel import paropen
from ase.units import Bohr, Hartree
def _read_dipole(self, out):
    """Reads total dipole moment from ONETEP output file"""
    line = ()
    while 'Total dipole moment' not in line:
        line = out.readline()
    dipolemoment = []
    for label, pos in sorted({'dx': 6, 'dy': 2, 'dz': 2}.items()):
        assert label in line.split()
        value = float(line.split()[pos]) * Bohr
        dipolemoment.append(value)
        line = out.readline()
    return array(dipolemoment)