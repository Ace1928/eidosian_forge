from copy import deepcopy
from os.path import isfile
from warnings import warn
from numpy import array
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.parallel import paropen
from ase.units import Bohr, Hartree
def _read_lattice(self, out):
    """ read the lattice parameters out of a onetep .out formatted file
        stream"""
    axes = []
    l = out.readline()
    conv_fac = Bohr
    if 'ang' in l:
        l = out.readline()
        conv_fac = 1.0
    elif 'bohr' in l:
        l = out.readline()
    for _ in range(0, 3):
        l = l.strip()
        p = l.split()
        if len(p) != 3:
            raise ReadError('Malformed Lattice block line "%s"' % l)
        try:
            axes.append([conv_fac * float(comp) for comp in p[0:3]])
        except ValueError:
            raise ReadError('Can\'t parse line "%s" in axes block' % l)
        l = out.readline()
    self.atoms.set_cell(axes)