from __future__ import annotations
import bisect
from copy import copy, deepcopy
from datetime import datetime
from math import log, pi, sqrt
from typing import TYPE_CHECKING, Any
from warnings import warn
import numpy as np
from monty.json import MSONable
from scipy import constants
from scipy.special import comb, erfc
from pymatgen.core.structure import Structure
from pymatgen.util.due import Doi, due
def _calc_ewald_terms(self):
    """Calculates and sets all Ewald terms (point, real and reciprocal)."""
    self._recip, recip_forces = self._calc_recip()
    self._real, self._point, real_point_forces = self._calc_real_and_point()
    if self._compute_forces:
        self._forces = recip_forces + real_point_forces