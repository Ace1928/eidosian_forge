import os
import copy
import subprocess
from math import pi, sqrt
import pathlib
from typing import Union, Optional, List, Set, Dict, Any
import warnings
import numpy as np
from ase.cell import Cell
from ase.outputs import Properties, all_outputs
from ase.utils import jsonable
from ase.calculators.abc import GetPropertiesMixin
def _deprecated_get_spin_polarized(self):
    msg = 'This calculator does not implement get_spin_polarized().  In the future, calc.get_spin_polarized() will work only on calculator classes that explicitly implement this method or inherit the method via specialized subclasses.'
    warnings.warn(msg, FutureWarning)
    return False