import re
import os
from ase.data import atomic_masses, atomic_numbers
from ase.calculators.lammpsrun import LAMMPS
from ase.calculators.lammpslib import LAMMPSlib
from ase.calculators.lammps import convert
from .kimmodel import KIMModelCalculator
from .exceptions import KIMCalculatorError
def _check_conflict_options(options, options_not_allowed, simulator):
    """Check whether options intended to be passed to a given calculator are allowed.
    Some options are not allowed because they must be set internally in this package.
    """
    s1 = set(options)
    s2 = set(options_not_allowed)
    common = s1.intersection(s2)
    if common:
        options_in_not_allowed = ', '.join(['"{}"'.format(s) for s in common])
        msg = 'Simulator "{}" does not support argument(s): {} provided in "options", because it is (they are) determined internally within the KIM calculator'.format(simulator, options_in_not_allowed)
        raise KIMCalculatorError(msg)