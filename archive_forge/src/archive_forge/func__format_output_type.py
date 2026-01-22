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
def _format_output_type(output_type):
    """ Given a letter: output_type, return
    a string formatted for a gaussian input file"""
    if output_type is None or output_type == '' or 't' in output_type.lower():
        output_type = 'P'
    return '#{}'.format(output_type)