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
@staticmethod
def _comps_from_fuzzy_formula(fuzzy_formula: str, m_dict: dict[str, float] | None=None, m_points: int=0, factor: float=1) -> Generator[tuple[Composition, int], None, None]:
    """A recursive helper method for formula parsing that helps in
        interpreting and ranking indeterminate formulas.
        Author: Anubhav Jain.

        Args:
            fuzzy_formula (str): A formula string, such as "co2o3" or "MN",
                that may or may not have multiple interpretations.
            m_dict (dict): A symbol:amt dictionary from the previously parsed
                formula.
            m_points: Number of points gained from the previously parsed
                formula.
            factor: Coefficient for this parse, e.g. (PO4)2 will feed in PO4
                as the fuzzy_formula with a coefficient of 2.

        Returns:
            list[tuple[Composition, int]]: A list of tuples, with the first element being a Composition
                and the second element being the number of points awarded that Composition interpretation.
        """
    m_dict = m_dict or {}

    def _parse_chomp_and_rank(m, f, m_dict, m_points):
        """A helper method for formula parsing that helps in interpreting and
            ranking indeterminate formulas
            Author: Anubhav Jain.

            Args:
                m: A regex match, with the first group being the element and
                    the second group being the amount
                f: The formula part containing the match
                m_dict: A symbol:amt dictionary from the previously parsed
                    formula
                m_points: Number of points gained from the previously parsed
                    formula

            Returns:
                A tuple of (f, m_dict, points) where m_dict now contains data
                from the match and the match has been removed (chomped) from
                the formula f. The "goodness" of the match determines the
                number of points returned for chomping. Returns
                (None, None, None) if no element could be found...
            """
        points = 0
        points_first_capital = 100
        points_second_lowercase = 100
        el = m.group(1)
        if len(el) > 2 or len(el) < 1:
            raise ValueError('Invalid element symbol entered!')
        amt = float(m.group(2)) if m.group(2).strip() != '' else 1
        char1 = el[0]
        char2 = el[1] if len(el) > 1 else ''
        if char1 == char1.upper():
            points += points_first_capital
        if char2 and char2 == char2.lower():
            points += points_second_lowercase
        el = char1.upper() + char2.lower()
        if Element.is_valid_symbol(el):
            if el in m_dict:
                m_dict[el] += amt * factor
            else:
                m_dict[el] = amt * factor
            return (f.replace(m.group(), '', 1), m_dict, m_points + points)
        return (None, None, None)
    fuzzy_formula = fuzzy_formula.strip()
    if len(fuzzy_formula) == 0:
        if m_dict:
            yield (Composition.from_dict(m_dict), m_points)
    else:
        for mp in re.finditer('\\(([^\\(\\)]+)\\)([\\.\\d]*)', fuzzy_formula):
            mp_points = m_points
            mp_form = fuzzy_formula.replace(mp.group(), ' ', 1)
            mp_dict = dict(m_dict)
            mp_factor = 1 if mp.group(2) == '' else float(mp.group(2))
            for match in Composition._comps_from_fuzzy_formula(mp.group(1), mp_dict, mp_points, factor=mp_factor):
                only_me = True
                for match2 in Composition._comps_from_fuzzy_formula(mp_form, mp_dict, mp_points, factor=1):
                    only_me = False
                    yield (match[0] + match2[0], match[1] + match2[1])
                if only_me:
                    yield match
            return
        m1 = re.match('([A-z])([\\.\\d]*)', fuzzy_formula)
        if m1:
            m_points1 = m_points
            m_form1 = fuzzy_formula
            m_dict1 = dict(m_dict)
            m_form1, m_dict1, m_points1 = _parse_chomp_and_rank(m1, m_form1, m_dict1, m_points1)
            if m_dict1:
                for match in Composition._comps_from_fuzzy_formula(m_form1, m_dict1, m_points1, factor):
                    yield match
        m2 = re.match('([A-z]{2})([\\.\\d]*)', fuzzy_formula)
        if m2:
            m_points2 = m_points
            m_form2 = fuzzy_formula
            m_dict2 = dict(m_dict)
            m_form2, m_dict2, m_points2 = _parse_chomp_and_rank(m2, m_form2, m_dict2, m_points2)
            if m_dict2:
                for match in Composition._comps_from_fuzzy_formula(m_form2, m_dict2, m_points2, factor):
                    yield match