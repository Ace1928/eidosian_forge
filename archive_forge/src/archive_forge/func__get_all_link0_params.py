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
def _get_all_link0_params(link0_section):
    """ Given a string link0_section which contains the link0
    section of a gaussian input file, returns a dictionary of
    keywords and values"""
    parameters = {}
    for line in link0_section:
        link0_match = _re_link0.match(line)
        link0_param = _get_link0_param(link0_match)
        if link0_param is not None:
            parameters.update(link0_param)
    return parameters