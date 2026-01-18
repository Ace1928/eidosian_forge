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
def zeroed(self, tol: float=0.001):
    """
        Args:
            tol: Tolerance.

        Returns:
            TensorCollection where small values are set to 0.
        """
    return type(self)([tensor.zeroed(tol) for tensor in self])