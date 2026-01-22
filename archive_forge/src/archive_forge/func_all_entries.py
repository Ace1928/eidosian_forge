from __future__ import annotations
import logging
import re
from itertools import chain, combinations
from typing import TYPE_CHECKING, no_type_check, overload
import numpy as np
from monty.fractions import gcd_float
from monty.json import MontyDecoder, MSONable
from uncertainties import ufloat
from pymatgen.core.composition import Composition
from pymatgen.entries.computed_entries import ComputedEntry
@property
def all_entries(self):
    """
        Equivalent of all_comp but returns entries, in the same order as the
        coefficients.
        """
    entries = []
    for comp in self._all_comp:
        for entry in self._all_entries:
            if entry.reduced_formula == comp.reduced_formula:
                entries.append(entry)
                break
    return entries