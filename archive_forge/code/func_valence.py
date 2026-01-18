from __future__ import annotations
import ast
import functools
import json
import re
import warnings
from collections import Counter
from enum import Enum, unique
from itertools import combinations, product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal
import numpy as np
from monty.json import MSONable
from pymatgen.core.units import SUPPORTED_UNIT_NAMES, FloatWithUnit, Ha_to_eV, Length, Mass, Unit
from pymatgen.util.string import Stringify, formula_double_format
@property
def valence(self):
    """From full electron config obtain valence subshell angular moment (L) and number of valence e- (v_e)."""
    if self.group == 18:
        return (np.nan, 0)
    L_symbols = 'SPDFGHIKLMNOQRTUVWXYZ'
    valence = []
    full_electron_config = self.full_electronic_structure
    last_orbital = full_electron_config[-1]
    for n, l_symbol, ne in full_electron_config:
        idx = L_symbols.lower().index(l_symbol)
        if ne < (2 * idx + 1) * 2 or ((n, l_symbol, ne) == last_orbital and ne == (2 * idx + 1) * 2 and (len(valence) == 0)):
            valence.append((idx, ne))
    if len(valence) > 1:
        raise ValueError(f'{self} has ambiguous valence')
    return valence[0]