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
def _get_atoms_info(line):
    """Returns the symbol and position of an atom from a line
    in the molecule specification section"""
    nuclear_props_match = _re_nuclear_props.search(line)
    if nuclear_props_match:
        line = line.replace(nuclear_props_match.group(0), '')
    tokens = line.split()
    symbol = _convert_to_symbol(tokens[0])
    pos = list(tokens[1:])
    return (symbol, pos)