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
def _pop_link0_params(params):
    """Takes the params dict and returns a dict with the link0 keywords
    removed, and a list containing the link0 keywords and options
    to be written to the gaussian input file"""
    params = deepcopy(params)
    out = []
    for key in _link0_keys:
        if key not in params:
            continue
        val = params.pop(key)
        if not val or (isinstance(val, str) and key.lower() == val.lower()):
            out.append('%{}'.format(key))
        else:
            out.append('%{}={}'.format(key, val))
    for key in _link0_special:
        if key not in params:
            continue
        val = params.pop(key)
        if not isinstance(val, str) and isinstance(val, Iterable):
            val = ' '.join(val)
        out.append('%{} L{}'.format(key, val))
    return (params, out)