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
def _format_route_params(params):
    """Get keywords and values from the params dictionary and return
    as a list of lines to add to the gaussian input file"""
    out = []
    for key, val in params.items():
        if not val or (isinstance(val, str) and key.lower() == val.lower()):
            out.append(key)
        elif not isinstance(val, str) and isinstance(val, Iterable):
            out.append('{}({})'.format(key, ','.join(val)))
        else:
            out.append('{}({})'.format(key, val))
    return out