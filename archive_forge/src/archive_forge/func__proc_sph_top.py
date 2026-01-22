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
def _proc_sph_top(self):
    """Handles Spherical Top Molecules, which belongs to the T, O or I point
        groups.
        """
    self._find_spherical_axes()
    if len(self.rot_sym) == 0:
        logger.debug('Accidental spherical top!')
        self._proc_sym_top()
    main_axis, rot = max(self.rot_sym, key=lambda v: v[1])
    if rot < 3:
        logger.debug('Accidental spherical top!')
        self._proc_sym_top()
    elif rot == 3:
        mirror_type = self._find_mirror(main_axis)
        if mirror_type != '':
            if self.is_valid_op(PointGroupAnalyzer.inversion_op):
                self.symmops.append(PointGroupAnalyzer.inversion_op)
                self.sch_symbol = 'Th'
            else:
                self.sch_symbol = 'Td'
        else:
            self.sch_symbol = 'T'
    elif rot == 4:
        if self.is_valid_op(PointGroupAnalyzer.inversion_op):
            self.symmops.append(PointGroupAnalyzer.inversion_op)
            self.sch_symbol = 'Oh'
        else:
            self.sch_symbol = 'O'
    elif rot == 5:
        if self.is_valid_op(PointGroupAnalyzer.inversion_op):
            self.symmops.append(PointGroupAnalyzer.inversion_op)
            self.sch_symbol = 'Ih'
        else:
            self.sch_symbol = 'I'