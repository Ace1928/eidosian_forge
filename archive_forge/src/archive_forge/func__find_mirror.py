from __future__ import annotations
import copy
import itertools
import logging
import math
import warnings
from collections import defaultdict
from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache
from math import cos, sin
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
import scipy.cluster
import spglib
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Molecule, PeriodicSite, Structure
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.util.coord import find_in_coord_list, pbc_diff
from pymatgen.util.due import Doi, due
def _find_mirror(self, axis):
    """Looks for mirror symmetry of specified type about axis.

        Possible types are "h" or "vd". Horizontal (h) mirrors are perpendicular to the
        axis while vertical (v) or diagonal (d) mirrors are parallel. v mirrors has atoms
        lying on the mirror plane while d mirrors do not.
        """
    mirror_type = ''
    if self.is_valid_op(SymmOp.reflection(axis)):
        self.symmops.append(SymmOp.reflection(axis))
        mirror_type = 'h'
    else:
        for s1, s2 in itertools.combinations(self.centered_mol, 2):
            if s1.species == s2.species:
                normal = s1.coords - s2.coords
                if np.dot(normal, axis) < self.tol:
                    op = SymmOp.reflection(normal)
                    if self.is_valid_op(op):
                        self.symmops.append(op)
                        if len(self.rot_sym) > 1:
                            mirror_type = 'd'
                            for v, _ in self.rot_sym:
                                if np.linalg.norm(v - axis) >= self.tol and np.dot(v, normal) < self.tol:
                                    mirror_type = 'v'
                                    break
                        else:
                            mirror_type = 'v'
                        break
    return mirror_type