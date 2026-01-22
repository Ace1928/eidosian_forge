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
def as_entry(self, energies) -> ComputedEntry:
    """
        Returns a ComputedEntry representation of the reaction.
        """
    relevant_comp = [comp * abs(coeff) for coeff, comp in zip(self._coeffs, self._all_comp)]
    comp: Composition = sum(relevant_comp, Composition())
    entry = ComputedEntry(0.5 * comp, self.calculate_energy(energies))
    entry.name = str(self)
    return entry