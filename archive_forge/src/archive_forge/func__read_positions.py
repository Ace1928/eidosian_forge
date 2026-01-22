from copy import deepcopy
from os.path import isfile
from warnings import warn
from numpy import array
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.parallel import paropen
from ase.units import Bohr, Hartree
def _read_positions(self, out):
    """Read the contents of a positions_abs block into the calculator's
        atoms object, setting both species and positions. Tries to strip out
        comment lines and is aware of angstom vs. bohr"""
    line = out.readline()
    conv_fac = Bohr
    if 'ang' in line:
        line = out.readline()
        conv_fac = 1.0
    elif 'bohr' in line:
        line = out.readline()
    symbols = []
    positions = []
    while '%endblock' not in line.lower():
        line = line.strip()
        if line[0] != '#':
            atom, suffix = line.split(None, 1)
            pos = suffix.split(None, 3)[0:3]
            try:
                pos = [conv_fac * float(p) for p in pos]
            except ValueError:
                raise ReadError('Malformed position line "%s"', line)
            symbols.append(atom)
            positions.append(pos)
        line = out.readline()
    tags = deepcopy(symbols)
    for j in range(len(symbols)):
        symbols[j] = ''.join((i for i in symbols[j] if not i.isdigit()))
    for j in range(len(tags)):
        tags[j] = ''.join((i for i in tags[j] if not i.isalpha()))
        if tags[j] == '':
            tags[j] = '0'
        tags[j] = int(tags[j])
    if len(self.atoms) != len(symbols):
        self.atoms = Atoms(symbols=symbols, positions=positions)
    self.atoms.set_chemical_symbols(symbols)
    self.atoms.set_tags(tags)
    self.atoms.set_positions(positions)