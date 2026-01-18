import functools
import json
import numbers
import operator
import os
import re
import warnings
from time import time
from typing import List, Dict, Any
import numpy as np
from ase.atoms import Atoms
from ase.calculators.calculator import all_properties, all_changes
from ase.data import atomic_numbers
from ase.db.row import AtomsRow
from ase.formula import Formula
from ase.io.jsonio import create_ase_object
from ase.parallel import world, DummyMPI, parallel_function, parallel_generator
from ase.utils import Lock, PurePath
def time_string_to_float(s):
    if isinstance(s, (float, int)):
        return s
    s = s.replace(' ', '')
    if '+' in s:
        return sum((time_string_to_float(x) for x in s.split('+')))
    if s[-2].isalpha() and s[-1] == 's':
        s = s[:-1]
    i = 1
    while s[i].isdigit():
        i += 1
    return seconds[s[i:]] * int(s[:i]) / YEAR