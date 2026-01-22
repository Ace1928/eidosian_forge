from collections import defaultdict
from itertools import chain, combinations, product, permutations
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.decorators import sympify_method_args, sympify_return
from sympy.core.function import Application, Derivative
from sympy.core.kind import BooleanKind, NumberKind
from sympy.core.numbers import Number
from sympy.core.operations import LatticeOp
from sympy.core.singleton import Singleton, S
from sympy.core.sorting import ordered
from sympy.core.sympify import _sympy_converter, _sympify, sympify
from sympy.utilities.iterables import sift, ibin
from sympy.utilities.misc import filldedent
def _convert_to_varsANF(term, variables):
    """
    Converts a term in the expansion of a function from binary to its
    variable form (for ANF).

    Parameters
    ==========

    term : list of 1's and 0's (complementation pattern)
    variables : list of variables

    """
    temp = [variables[n] for n, t in enumerate(term) if t == 1]
    if not temp:
        return true
    return And(*temp)