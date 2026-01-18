import re
from collections import OrderedDict
import numpy as np
from ase import Atoms
from ase.units import Hartree, Bohr
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
from .parser import _define_pattern
Splits an NWChem output file into chunks corresponding to
    individual single point calculations.