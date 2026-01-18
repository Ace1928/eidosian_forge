import os
import warnings
import time
from typing import Optional
import re
import numpy as np
from ase.units import Hartree
from ase.io.aims import write_aims, read_aims
from ase.data import atomic_numbers
from ase.calculators.calculator import FileIOCalculator, Parameters, kpts2mp, \
def translate_tier(self, tier):
    if tier.lower() == 'first':
        return 1
    elif tier.lower() == 'second':
        return 2
    elif tier.lower() == 'third':
        return 3
    elif tier.lower() == 'fourth':
        return 4
    else:
        return -1