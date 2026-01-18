import os
import sys
import errno
import pickle
import warnings
import collections
import numpy as np
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.calculator import PropertyNotImplementedError
from ase.constraints import FixAtoms
from ase.parallel import world, barrier
def unstringnify_info(stringnified):
    """Convert the dict *stringnified* to a dict with unstringnified
    objects and return it.  Objects that cannot be unpickled will be
    skipped and a warning will be issued."""
    info = {}
    for k, s in stringnified.items():
        try:
            v = pickle.loads(s)
        except pickle.UnpicklingError:
            warnings.warn('Skipping not unpicklable info-dict item: ' + '"%s" (%s)' % (k, sys.exc_info()[1]), UserWarning)
        else:
            info[k] = v
    return info