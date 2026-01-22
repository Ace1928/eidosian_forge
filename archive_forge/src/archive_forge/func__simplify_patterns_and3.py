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
def _simplify_patterns_and3():
    """ Three-term patterns for And."""
    from sympy.core import Wild
    from sympy.core.relational import Eq, Ge, Gt
    a = Wild('a')
    b = Wild('b')
    c = Wild('c')
    _matchers_and = ((Tuple(Ge(a, b), Ge(b, c), Gt(c, a)), false), (Tuple(Ge(a, b), Gt(b, c), Gt(c, a)), false), (Tuple(Gt(a, b), Gt(b, c), Gt(c, a)), false), (Tuple(Ge(a, b), Ge(a, c), Ge(b, c)), And(Ge(a, b), Ge(b, c))), (Tuple(Ge(a, b), Ge(a, c), Gt(b, c)), And(Ge(a, b), Gt(b, c))), (Tuple(Ge(a, b), Gt(a, c), Gt(b, c)), And(Ge(a, b), Gt(b, c))), (Tuple(Ge(a, c), Gt(a, b), Gt(b, c)), And(Gt(a, b), Gt(b, c))), (Tuple(Ge(b, c), Gt(a, b), Gt(a, c)), And(Gt(a, b), Ge(b, c))), (Tuple(Gt(a, b), Gt(a, c), Gt(b, c)), And(Gt(a, b), Gt(b, c))), (Tuple(Ge(b, a), Ge(c, a), Ge(b, c)), And(Ge(c, a), Ge(b, c))), (Tuple(Ge(b, a), Ge(c, a), Gt(b, c)), And(Ge(c, a), Gt(b, c))), (Tuple(Ge(b, a), Gt(c, a), Gt(b, c)), And(Gt(c, a), Gt(b, c))), (Tuple(Ge(c, a), Gt(b, a), Gt(b, c)), And(Ge(c, a), Gt(b, c))), (Tuple(Ge(b, c), Gt(b, a), Gt(c, a)), And(Gt(c, a), Ge(b, c))), (Tuple(Gt(b, a), Gt(c, a), Gt(b, c)), And(Gt(c, a), Gt(b, c))), (Tuple(Ge(a, b), Ge(b, c), Ge(c, a)), And(Eq(a, b), Eq(b, c))))
    return _matchers_and