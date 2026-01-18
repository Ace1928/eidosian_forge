from copy import deepcopy
from os.path import isfile
from warnings import warn
from numpy import array
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.parallel import paropen
from ase.units import Bohr, Hartree
def write_kpt_path(self, fd, path):
    """Writes a k-point path to a ONETEP input file"""
    for kpt in array(path):
        fd.write('    %8.6f %8.6f %8.6f\n' % (kpt[0], kpt[1], kpt[2]))