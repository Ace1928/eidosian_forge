from __future__ import annotations
import collections
import itertools
import os
import string
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from monty.serialization import loadfn
from scipy.linalg import polar
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def _get_item_index(self, item):
    if len(self._tensor_list) == 0:
        return None
    item = np.array(item)
    axis = tuple(range(1, len(item.shape) + 1))
    mask = np.all(np.abs(np.array(self._tensor_list) - item) < self.tol, axis=axis)
    indices = np.where(mask)[0]
    if len(indices) > 1:
        raise ValueError('Tensor key collision.')
    if len(indices) == 0:
        return None
    return indices[0]