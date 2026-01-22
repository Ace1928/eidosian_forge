from typing import Tuple as tTuple
from collections import defaultdict
from functools import cmp_to_key, reduce
from itertools import product
import operator
from .sympify import sympify
from .basic import Basic
from .singleton import S
from .operations import AssocOp, AssocOpDispatcher
from .cache import cacheit
from .logic import fuzzy_not, _fuzzy_group
from .expr import Expr
from .parameters import global_parameters
from .kind import KindDispatcher
from .traversal import bottom_up
from sympy.utilities.iterables import sift
from .numbers import Rational
from .power import Pow
from .add import Add, _unevaluated_Add
def _handle_for_oo(c_part, coeff_sign):
    new_c_part = []
    for t in c_part:
        if t.is_extended_positive:
            continue
        if t.is_extended_negative:
            coeff_sign *= -1
            continue
        new_c_part.append(t)
    return (new_c_part, coeff_sign)