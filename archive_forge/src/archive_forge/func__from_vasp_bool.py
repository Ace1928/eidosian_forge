import os
import warnings
import shutil
from os.path import join, isfile, islink
from typing import List, Sequence, Tuple
import numpy as np
import ase
from ase.calculators.calculator import kpts2ndarray
from ase.calculators.vasp.setups import get_default_setups
def _from_vasp_bool(x):
    """Cast vasp boolean to Python bool

    VASP files sometimes use T or F as shorthand for the preferred Boolean
    notation .TRUE. or .FALSE. As capitalisation is pretty inconsistent in
    practice, we allow all cases to be cast to a Python bool.

    """
    assert isinstance(x, str)
    if x.lower() == '.true.' or x.lower() == 't':
        return True
    elif x.lower() == '.false.' or x.lower() == 'f':
        return False
    else:
        raise ValueError('Value "%s" not recognized as bool' % x)