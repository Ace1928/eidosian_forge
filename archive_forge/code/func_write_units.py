import re
import numpy as np
from collections import OrderedDict
import ase.units
from ase.atoms import Atoms
from ase.spacegroup import Spacegroup
from ase.spacegroup.spacegroup import SpacegroupNotFoundError
from ase.calculators.singlepoint import SinglePointDFTCalculator
def write_units(data, out):
    if 'units' in data:
        for tag, units in data['units']:
            out.append('  units %s %s' % (tag, units))