from copy import deepcopy
from os.path import isfile
from warnings import warn
from numpy import array
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.parallel import paropen
from ase.units import Bohr, Hartree
def write_groups(self, fd, groups):
    """Writes multiple groups of atom labels to a ONETEP input file"""
    for grp in groups:
        fd.write(' '.join(map(str, grp)))
        fd.write('\n')