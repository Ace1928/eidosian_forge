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
def _parse_formula(self, formula: str, strict: bool=True) -> dict[str, float]:
    """
        Args:
            formula (str): A string formula, e.g. Fe2O3, Li3Fe2(PO4)3.
            strict (bool): Whether to throw an error if formula string is invalid (e.g. empty).
                Defaults to True.

        Returns:
            Composition with that formula.

        Notes:
            In the case of Metallofullerene formula (e.g. Y3N@C80),
            the @ mark will be dropped and passed to parser.
        """
    if strict and re.match('[\\s\\d.*/]*$', formula):
        raise ValueError(f'Invalid formula={formula!r}')
    formula = formula.replace('@', '')
    formula = formula.replace('[', '(')
    formula = formula.replace(']', ')')

    def get_sym_dict(form: str, factor: float) -> dict[str, float]:
        sym_dict: dict[str, float] = collections.defaultdict(float)
        for m in re.finditer('([A-Z][a-z]*)\\s*([-*\\.e\\d]*)', form):
            el = m.group(1)
            amt = 1.0
            if m.group(2).strip() != '':
                amt = float(m.group(2))
            sym_dict[el] += amt * factor
            form = form.replace(m.group(), '', 1)
        if form.strip():
            raise ValueError(f'{form} is an invalid formula!')
        return sym_dict
    match = re.search('\\(([^\\(\\)]+)\\)\\s*([\\.e\\d]*)', formula)
    while match:
        factor = 1.0
        if match.group(2) != '':
            factor = float(match.group(2))
        unit_sym_dict = get_sym_dict(match.group(1), factor)
        expanded_sym = ''.join((f'{el}{amt}' for el, amt in unit_sym_dict.items()))
        expanded_formula = formula.replace(match.group(), expanded_sym, 1)
        formula = expanded_formula
        match = re.search('\\(([^\\(\\)]+)\\)\\s*([\\.e\\d]*)', formula)
    return get_sym_dict(formula, 1)