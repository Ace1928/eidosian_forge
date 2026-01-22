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
def _fetch_pspots(self, directory=None):
    """Put all specified pseudo-potentials into the working directory.
        """
    if not os.environ.get('PSPOT_DIR', None) and self._castep_pp_path == os.path.abspath('.'):
        return
    if directory is None:
        directory = self._directory
    if not os.path.isdir(self._castep_pp_path):
        warnings.warn('PSPs directory %s not found' % self._castep_pp_path)
    pspots = {}
    if self._find_pspots:
        self.find_pspots()
    if self.cell.species_pot.value is not None:
        for line in self.cell.species_pot.value.split('\n'):
            line = line.split()
            if line:
                pspots[line[0]] = line[1]
    for species in self.atoms.get_chemical_symbols():
        if not pspots or species not in pspots.keys():
            if self._build_missing_pspots:
                if self._pedantic:
                    warnings.warn('Warning: you have no PP specified for %s. CASTEP will now generate an on-the-fly potentials. For sake of numerical consistency and efficiency this is discouraged.' % species)
            else:
                raise RuntimeError('Warning: you have no PP specified for %s.' % species)
    if self.cell.species_pot.value:
        for species, pspot in pspots.items():
            orig_pspot_file = os.path.join(self._castep_pp_path, pspot)
            cp_pspot_file = os.path.join(directory, pspot)
            if os.path.exists(orig_pspot_file) and (not os.path.exists(cp_pspot_file)):
                if self._copy_pspots:
                    shutil.copy(orig_pspot_file, directory)
                elif self._link_pspots:
                    os.symlink(orig_pspot_file, cp_pspot_file)
                elif self._pedantic:
                    warnings.warn('Warning: PP files have neither been linked nor copied to the working directory. Make sure to set the evironment variable PSPOT_DIR accordingly!')