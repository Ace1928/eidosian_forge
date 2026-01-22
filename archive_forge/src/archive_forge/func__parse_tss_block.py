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
def _parse_tss_block(value, scaled=False):
    is_atoms = isinstance(value, ase.atoms.Atoms)
    try:
        is_strlist = all(map(lambda x: isinstance(x, str), value))
    except TypeError:
        is_strlist = False
    if not is_atoms:
        if not is_strlist:
            raise TypeError('castep.cell.positions_abs/frac_intermediate/product expects Atoms object or list of strings')
        if not scaled and value[0].strip() != 'ang':
            raise RuntimeError('Only ang units currently supported in castep.cell.positions_abs_intermediate/product')
        return '\n'.join(map(str.strip, value))
    else:
        text_block = '' if scaled else 'ang\n'
        positions = value.get_scaled_positions() if scaled else value.get_positions()
        symbols = value.get_chemical_symbols()
        for s, p in zip(symbols, positions):
            text_block += '    {0} {1:.3f} {2:.3f} {3:.3f}\n'.format(s, *p)
        return text_block