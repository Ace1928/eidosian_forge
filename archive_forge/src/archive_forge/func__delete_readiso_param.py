import re
import warnings
from collections.abc import Iterable
from copy import deepcopy
import numpy as np
from ase import Atoms
from ase.calculators.calculator import InputError, Calculator
from ase.calculators.gaussian import Gaussian
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import atomic_masses_iupac2016, chemical_symbols
from ase.io import ParseError
from ase.io.zmatrix import parse_zmatrix
from ase.units import Bohr, Hartree
def _delete_readiso_param(parameters):
    """Removes the readiso parameter from the parameters dict"""
    parameters = deepcopy(parameters)
    freq_name, freq_options = _get_readiso_param(parameters)
    if freq_name is not None:
        if 'readisotopes' in freq_options:
            iso_name = 'readisotopes'
        else:
            iso_name = 'readiso'
        freq_options = [v.group() for v in re.finditer('[^\\,/\\s]+', freq_options)]
        freq_options.remove(iso_name)
        new_freq_options = ''
        for v in freq_options:
            new_freq_options += v + ' '
        if new_freq_options == '':
            new_freq_options = None
        else:
            new_freq_options = new_freq_options.strip()
        parameters[freq_name] = new_freq_options
    return parameters