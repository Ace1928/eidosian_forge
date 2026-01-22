import difflib
import numpy as np
import os
import re
import glob
import shutil
import sys
import json
import time
import tempfile
import warnings
import subprocess
from copy import deepcopy
from collections import namedtuple
from itertools import product
from typing import List, Set
import ase
import ase.units as units
from ase.calculators.general import Calculator
from ase.calculators.calculator import compare_atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.calculator import kpts2sizeandoffsets
from ase.dft.kpoints import BandPath
from ase.parallel import paropen
from ase.io.castep import read_param
from ase.io.castep import read_bands
from ase.constraints import FixCartesian
def _get_absolute_number(self, species, nic, atoms=None):
    """This is the inverse function to _get_number in species."""
    if atoms is None:
        atoms = self.atoms
    ch = atoms.get_chemical_symbols()
    ch.reverse()
    total_nr = 0
    assert nic > 0, 'Number in species needs to be 1 or larger'
    while True:
        if ch.pop() == species:
            if nic == 1:
                return total_nr
            nic -= 1
        total_nr += 1