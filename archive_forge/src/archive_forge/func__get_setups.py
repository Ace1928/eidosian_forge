import os
import warnings
import shutil
from os.path import join, isfile, islink
from typing import List, Sequence, Tuple
import numpy as np
import ase
from ase.calculators.calculator import kpts2ndarray
from ase.calculators.vasp.setups import get_default_setups
def _get_setups(self):
    p = self.input_params
    special_setups = []
    setups_defaults = get_default_setups()
    if p['setups'] is None:
        p['setups'] = {'base': 'minimal'}
    elif isinstance(p['setups'], str):
        if p['setups'].lower() in setups_defaults.keys():
            p['setups'] = {'base': p['setups']}
    if 'base' in p['setups']:
        setups = setups_defaults[p['setups']['base'].lower()]
    else:
        setups = {}
    if p['setups'] is not None:
        setups.update(p['setups'])
    for m in setups:
        try:
            special_setups.append(int(m))
        except ValueError:
            pass
    return (setups, special_setups)