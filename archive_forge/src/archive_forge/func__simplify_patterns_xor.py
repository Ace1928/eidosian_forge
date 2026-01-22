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
@cacheit
def _simplify_patterns_xor():
    """ Two-term patterns for Xor."""
    from sympy.functions.elementary.miscellaneous import Min, Max
    from sympy.core import Wild
    from sympy.core.relational import Eq, Ne, Ge, Gt, Le, Lt
    a = Wild('a')
    b = Wild('b')
    c = Wild('c')
    _matchers_xor = ((Tuple(Eq(a, b), Le(a, b)), Lt(a, b)), (Tuple(Eq(a, b), Lt(a, b)), Le(a, b)), (Tuple(Le(a, b), Lt(a, b)), Eq(a, b)), (Tuple(Le(a, b), Le(b, a)), Ne(a, b)), (Tuple(Le(b, a), Ne(a, b)), Le(a, b)), (Tuple(Lt(b, a), Ne(a, b)), Lt(a, b)), (Tuple(Le(b, a), Le(c, a)), And(Ge(a, Min(b, c)), Lt(a, Max(b, c)))), (Tuple(Le(b, a), Lt(c, a)), ITE(b > c, And(Gt(a, c), Lt(a, b)), And(Ge(a, b), Le(a, c)))), (Tuple(Lt(b, a), Lt(c, a)), And(Gt(a, Min(b, c)), Le(a, Max(b, c)))), (Tuple(Le(a, b), Le(a, c)), And(Le(a, Max(b, c)), Gt(a, Min(b, c)))), (Tuple(Le(a, b), Lt(a, c)), ITE(b < c, And(Lt(a, c), Gt(a, b)), And(Le(a, b), Ge(a, c)))), (Tuple(Lt(a, b), Lt(a, c)), And(Lt(a, Max(b, c)), Ge(a, Min(b, c)))))
    return _matchers_xor