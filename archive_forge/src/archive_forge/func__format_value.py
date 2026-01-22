import os
import re
from subprocess import call, TimeoutExpired
from copy import deepcopy
import numpy as np
from ase import Atoms
from ase.utils import workdir
from ase.units import Hartree, Bohr, Debye
from ase.calculators.singlepoint import SinglePointCalculator
def _format_value(val):
    if isinstance(val, bool):
        return '.t.' if val else '.f.'
    return str(val).upper()