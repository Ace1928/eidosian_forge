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
def _validate_params(parameters):
    """Checks whether all of the required parameters exist in the
    parameters dict and whether it contains any unsupported settings
    """
    unsupported_settings = {'z-matrix', 'modredun', 'modredundant', 'addredundant', 'addredun', 'readopt', 'rdopt'}
    for s in unsupported_settings:
        for v in parameters.values():
            if v is not None and s in str(v):
                raise ParseError('ERROR: Could not read the Gaussian input file, as the option: {} is currently unsupported.'.format(s))
    for k in list(parameters.keys()):
        if 'popt' in k:
            parameters['opt'] = parameters.pop(k)
            warnings.warn('The option {} is currently unsupported. This has been replaced with {}.'.format('POpt', 'opt'))
            return