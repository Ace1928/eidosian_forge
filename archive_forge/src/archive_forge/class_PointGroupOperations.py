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
class PointGroupOperations(list):
    """Defines a point group, which is essentially a sequence of symmetry operations.

    Attributes:
        sch_symbol (str): Schoenflies symbol of the point group.
    """

    def __init__(self, sch_symbol, operations, tol: float=0.1):
        """
        Args:
            sch_symbol (str): Schoenflies symbol of the point group.
            operations ([SymmOp]): Initial set of symmetry operations. It is
                sufficient to provide only just enough operations to generate
                the full set of symmetries.
            tol (float): Tolerance to generate the full set of symmetry
                operations.
        """
        self.sch_symbol = sch_symbol
        super().__init__(generate_full_symmops(operations, tol))

    def __repr__(self):
        return self.sch_symbol