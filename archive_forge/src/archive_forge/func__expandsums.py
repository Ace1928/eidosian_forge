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
@staticmethod
def _expandsums(sums):
    """
        Helper function for _eval_expand_mul.

        sums must be a list of instances of Basic.
        """
    L = len(sums)
    if L == 1:
        return sums[0].args
    terms = []
    left = Mul._expandsums(sums[:L // 2])
    right = Mul._expandsums(sums[L // 2:])
    terms = [Mul(a, b) for a in left for b in right]
    added = Add(*terms)
    return Add.make_args(added)