from __future__ import annotations
import abc
import itertools
from math import ceil, cos, e, pi, sin, tan
from typing import TYPE_CHECKING, Any
from warnings import warn
import networkx as nx
import numpy as np
import spglib
from monty.dev import requires
from scipy.linalg import sqrtm
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import MagSymmOp, SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, cite_conventional_cell_algo
@staticmethod
def _closewrapped(pos1, pos2, tolerance):
    pos1 = pos1 % 1.0
    pos2 = pos2 % 1.0
    if len(pos1) != len(pos2):
        return False
    for idx, p1 in enumerate(pos1):
        if abs(p1 - pos2[idx]) > tolerance[idx] and abs(p1 - pos2[idx]) < 1 - tolerance[idx]:
            return False
    return True