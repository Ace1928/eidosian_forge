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
@classmethod
def _str_from_comp(cls, coeffs, compositions, reduce=False) -> tuple[str, float]:
    r_coeffs = np.zeros(len(coeffs))
    r_formulas = []
    for idx, (amt, comp) in enumerate(zip(coeffs, compositions)):
        formula, factor = comp.get_reduced_formula_and_factor()
        r_coeffs[idx] = amt * factor
        r_formulas.append(formula)
    if reduce:
        factor = 1 / gcd_float(np.abs(r_coeffs))
        r_coeffs *= factor
    else:
        factor = 1
    return (cls._str_from_formulas(r_coeffs, r_formulas), factor)