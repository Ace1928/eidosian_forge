import sys
import time
import copy
import warnings
from abc import ABC, abstractmethod
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline
from ase.constraints import Filter, FixAtoms
from ase.utils import longsum
from ase.geometry import find_mic
import ase.utils.ff as ff
import ase.units as units
from ase.optimize.precon.neighbors import (get_neighbours,
from ase.neighborlist import neighbor_list
def apply_fixed(atoms, P):
    fixed_atoms = []
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            fixed_atoms.extend(list(constraint.index))
        else:
            raise TypeError('only FixAtoms constraints are supported by Precon class')
    if len(fixed_atoms) != 0:
        P = P.tolil()
    for i in fixed_atoms:
        P[i, :] = 0.0
        P[:, i] = 0.0
        P[i, i] = 1.0
    return P