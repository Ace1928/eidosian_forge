import re
import numpy as np
from collections import OrderedDict
import ase.units
from ase.atoms import Atoms
from ase.spacegroup import Spacegroup
from ase.spacegroup.spacegroup import SpacegroupNotFoundError
from ase.calculators.singlepoint import SinglePointDFTCalculator
def write_generic_block(data):
    out = []
    for tag, data in data.items():
        for value in data:
            out.append('%s %s' % (tag, ' '.join((str(x) for x in value))))
    return '\n'.join(out)