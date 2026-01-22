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
def _compare_merge_configs(configs, new):
    """Append new to configs if it contains a new geometry or new data.

    Gaussian sometimes repeats a geometry, for example at the end of an
    optimization, or when a user requests vibrational frequency
    analysis in the same calculation as a geometry optimization.

    In those cases, rather than repeating the structure in the list of
    returned structures, try to merge results if doing so doesn't change
    any previously calculated values. If that's not possible, then create
    a new "image" with the new results.
    """
    if not configs:
        configs.append(new)
        return
    old = configs[-1]
    if old != new:
        configs.append(new)
        return
    oldres = old.calc.results
    newres = new.calc.results
    common_keys = set(oldres).intersection(newres)
    for key in common_keys:
        if np.any(oldres[key] != newres[key]):
            configs.append(new)
            return
    else:
        oldres.update(newres)