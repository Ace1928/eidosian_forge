import os
import warnings
import shutil
from os.path import join, isfile, islink
from typing import List, Sequence, Tuple
import numpy as np
import ase
from ase.calculators.calculator import kpts2ndarray
from ase.calculators.vasp.setups import get_default_setups
def _to_vasp_bool(x):
    """Convert Python boolean to string for VASP input

    In case the value was modified to a string already, appropriate strings
    will also be accepted and cast to a standard .TRUE. / .FALSE. format.

    """
    if isinstance(x, str):
        if x.lower() in ('.true.', 't'):
            x = True
        elif x.lower() in ('.false.', 'f'):
            x = False
        else:
            raise ValueError('"%s" not recognised as VASP Boolean')
    assert isinstance(x, bool)
    if x:
        return '.TRUE.'
    else:
        return '.FALSE.'