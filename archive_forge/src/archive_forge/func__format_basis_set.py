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
def _format_basis_set(basis, basisfile, basis_set):
    """Format either: the basis set filename (basisfile), the basis set file
    contents (from reading basisfile), or the basis_set text as a list of
    strings to be added to the gaussian input file."""
    out = []
    if basisfile is not None:
        if basisfile[0] == '@':
            out.append(basisfile)
        else:
            with open(basisfile, 'r') as fd:
                out.append(fd.read())
    elif basis_set is not None:
        out.append(basis_set)
    elif basis is not None and basis.lower() == 'gen':
        raise InputError('Please set basisfile or basis_set')
    return out