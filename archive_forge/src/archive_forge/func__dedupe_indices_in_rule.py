from __future__ import annotations
from typing import Any
from functools import reduce
from math import prod
from abc import abstractmethod, ABC
from collections import defaultdict
import operator
import itertools
from sympy.core.numbers import (Integer, Rational)
from sympy.combinatorics import Permutation
from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, \
from sympy.core import Basic, Expr, sympify, Add, Mul, S
from sympy.core.containers import Tuple, Dict
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import CantSympify, _sympify
from sympy.core.operations import AssocOp
from sympy.external.gmpy import SYMPY_INTS
from sympy.matrices import eye
from sympy.utilities.exceptions import (sympy_deprecation_warning,
from sympy.utilities.decorator import memoize_property, deprecated
from sympy.utilities.iterables import sift
def _dedupe_indices_in_rule(self, rule):
    """
        rule: dict

        This applies TensMul._dedupe_indices on all values of rule.

        """
    index_rules = {k: v for k, v in rule.items() if isinstance(k, TensorIndex)}
    other_rules = {k: v for k, v in rule.items() if k not in index_rules.keys()}
    exclude = set(self.get_indices())
    newrule = {}
    newrule.update(index_rules)
    exclude.update(index_rules.keys())
    exclude.update(index_rules.values())
    for old, new in other_rules.items():
        new_renamed = TensMul._dedupe_indices(new, exclude)
        if old == new or new_renamed is None:
            newrule[old] = new
        else:
            newrule[old] = new_renamed
            exclude.update(get_indices(new_renamed))
    return newrule