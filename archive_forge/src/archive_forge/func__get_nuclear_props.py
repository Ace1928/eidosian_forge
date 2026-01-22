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
def _get_nuclear_props(line):
    """ Reads any info in parantheses in the line and returns
    a dictionary of the nuclear properties."""
    nuclear_props_match = _re_nuclear_props.search(line)
    nuclear_props = {}
    if nuclear_props_match:
        nuclear_props = _get_key_value_pairs(nuclear_props_match.group(1))
        updated_nuclear_props = {}
        for key, value in nuclear_props.items():
            if value.isnumeric():
                value = int(value)
            else:
                value = float(value)
            if key not in _nuclear_prop_names:
                if 'fragment' in key:
                    warnings.warn('Fragments are not currently supported.')
                warnings.warn('The following nuclear properties could not be saved: {}'.format({key: value}))
            else:
                updated_nuclear_props[key] = value
        nuclear_props = updated_nuclear_props
    for k in _nuclear_prop_names:
        if k not in nuclear_props:
            nuclear_props[k] = None
    return nuclear_props