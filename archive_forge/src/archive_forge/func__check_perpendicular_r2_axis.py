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
def _check_perpendicular_r2_axis(self, axis):
    """Checks for R2 axes perpendicular to unique axis.

        For handling symmetric top molecules.
        """
    min_set = self._get_smallest_set_not_on_axis(axis)
    for s1, s2 in itertools.combinations(min_set, 2):
        test_axis = np.cross(s1.coords - s2.coords, axis)
        if np.linalg.norm(test_axis) > self.tol:
            op = SymmOp.from_axis_angle_and_translation(test_axis, 180)
            r2present = self.is_valid_op(op)
            if r2present:
                self.symmops.append(op)
                self.rot_sym.append((test_axis, 2))
                return True
    return None