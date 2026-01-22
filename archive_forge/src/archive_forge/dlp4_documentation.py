import re
from numpy import zeros, isscalar
from ase.atoms import Atoms
from ase.units import _auf, _amu, _auv
from ase.data import chemical_symbols
from ase.calculators.singlepoint import SinglePointCalculator
Write a DL_POLY_4 config file.

    Typically used indirectly through write('filename', atoms, format='dlp4').

    Can be unforgiven with custom chemical element names.
    Please complain to alin@elena.space in case of bugs