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
def _get_readiso_param(parameters):
    """ Returns a tuple containing the frequency
    keyword and its options, if the frequency keyword is
    present in parameters and ReadIso is one of its options"""
    freq_options = parameters.get('freq', None)
    if freq_options:
        freq_name = 'freq'
    else:
        freq_options = parameters.get('frequency', None)
        freq_name = 'frequency'
    if freq_options is not None:
        if ('readiso' or 'readisotopes') in freq_options:
            return (freq_name, freq_options)
    return (None, None)