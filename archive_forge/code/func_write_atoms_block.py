import re
import numpy as np
from collections import OrderedDict
import ase.units
from ase.atoms import Atoms
from ase.spacegroup import Spacegroup
from ase.spacegroup.spacegroup import SpacegroupNotFoundError
from ase.calculators.singlepoint import SinglePointDFTCalculator
def write_atoms_block(data):
    out = []
    write_units(data, out)
    if 'lattice' in data:
        for lat in data['lattice']:
            out.append('  lattice %s' % tensor_string(lat))
    if 'symmetry' in data:
        for sym in data['symmetry']:
            out.append('  symmetry %s' % sym)
    if 'atom' in data:
        for a in data['atom']:
            out.append('  atom %s %s %s %s' % (a['species'], a['label'], a['index'], ' '.join((str(x) for x in a['position']))))
    return '\n'.join(out)