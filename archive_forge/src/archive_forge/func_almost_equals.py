from __future__ import annotations
import collections
import os
import re
import string
import warnings
from functools import total_ordering
from itertools import combinations_with_replacement, product
from math import isnan
from typing import TYPE_CHECKING, cast
from monty.fractions import gcd, gcd_float
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.core.periodic_table import DummySpecies, Element, ElementType, Species, get_el_sp
from pymatgen.core.units import Mass
from pymatgen.util.string import Stringify, formula_double_format
def almost_equals(self, other: Composition, rtol: float=0.1, atol: float=1e-08) -> bool:
    """Returns true if compositions are equal within a tolerance.

        Args:
            other (Composition): Other composition to check
            rtol (float): Relative tolerance
            atol (float): Absolute tolerance
        """
    sps = set(self.elements + other.elements)
    for sp in sps:
        a = self[sp]
        b = other[sp]
        tol = atol + rtol * (abs(a) + abs(b)) / 2
        if abs(b - a) > tol:
            return False
    return True